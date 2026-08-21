// Native OpenFST wrapper for speech_to_phrase.
//
// Built against the CPython *limited* API (Py_LIMITED_API) so a single wheel
// works across Python point releases (abi3). All FST work is in-memory: this
// replaces the old subprocess + text-file pipeline.
//
// Label convention: a tokenizer token id `t` (>= 0) maps to OpenFST arc label
// `t + 1`; epsilon is 0. The CTC blank (id == num_tokens) maps to `num_tokens+1`
// on the input side and is consumed to epsilon on the output side.
//
// Pruning: the decode lattice only needs arcs for tokens the grammar can accept
// (plus blank). Any other acoustic token fails to compose, so dropping it is
// exact -- this prunes the per-frame branching from the full vocab (~1000) to
// the grammar's token set (~100) with no accuracy loss.
//
// Exposed functions:
//   build_grammar(num_tokens, blank_id, arcs_i32, finals_i32) -> capsule
//   save_grammar(capsule, path) / load_grammar(path) -> capsule
//   decode(capsule, logprobs_f32, T, V, blank_id, beam, token_bonus=0.0)
//        -> (token_ids:list[int], best_cost:float|None, second_cost:float|None)
//
// token_bonus is a word-insertion reward: `token_bonus` is subtracted from the
// cost of each emitting arc so longer, well-fitting parses compete fairly with
// short ones (plain CTC blanks are near-free, so the shortest parse otherwise
// wins regardless of acoustic fit). It affects path *selection* only -- the
// returned costs are the true acoustic costs (the bonus is added back), so the
// per-token score and its gate keep their meaning. 0 disables it (default).

#define PY_SSIZE_T_CLEAN
#ifndef Py_LIMITED_API
#define Py_LIMITED_API 0x030C0000  // CPython 3.12+
#endif
#include <Python.h>

#include <algorithm>
#include <cstdint>
#include <fstream>
#include <limits>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include <fst/fstlib.h>

using fst::StdArc;
using fst::StdVectorFst;
using StateId = StdArc::StateId;
using Weight = StdArc::Weight;

static const char *CAPSULE_NAME = "speech_to_phrase._fst.Grammar";

// A compiled grammar: the token->sentence FST plus the set of acoustic token
// ids it can accept (for exact decode-lattice pruning).
struct Grammar {
  StdVectorFst *fst = nullptr;
  std::vector<int> tokens;  // grammar token ids (excludes blank)
  ~Grammar() { delete fst; }
};

// ----------------------------------------------------------------------------
// FST construction
// ----------------------------------------------------------------------------

// CTC topology mapping an acoustic token sequence (with blank + repeats) to the
// collapsed token sequence, restricted to the tokens the grammar uses.
static StdVectorFst *build_token2char(const std::vector<int> &tokens,
                                      int blank_label) {
  auto *f = new StdVectorFst();
  StateId start = f->AddState();
  f->SetStart(start);
  f->SetFinal(start, Weight::One());

  std::unordered_map<int, StateId> tok_state;
  tok_state.reserve(tokens.size() * 2);
  for (int t : tokens) {
    StateId s = f->AddState();
    f->SetFinal(s, Weight::One());
    tok_state[t] = s;
  }

  // Blank self-loop at start (consume blank, emit nothing).
  f->AddArc(start, StdArc(blank_label, 0, Weight::One(), start));

  for (int t : tokens) {
    const int tl = t + 1;
    const StateId s = tok_state[t];
    f->AddArc(start, StdArc(tl, tl, Weight::One(), s));   // first occurrence emits
    f->AddArc(s, StdArc(tl, 0, Weight::One(), s));        // repeat -> epsilon
    f->AddArc(s, StdArc(blank_label, 0, Weight::One(), start));  // blank resets
    for (int u : tokens) {                                // different token emits now
      if (u == t) continue;
      const int ul = u + 1;
      f->AddArc(s, StdArc(ul, ul, Weight::One(), tok_state[u]));
    }
  }
  return f;
}

// Grammar acceptor from templates.py arcs: input==output==token-id label.
static StdVectorFst *build_char2sen(const int32_t *arcs, Py_ssize_t n_arcs,
                                    const int32_t *finals, Py_ssize_t n_finals,
                                    std::unordered_set<int> *used_tokens) {
  int max_state = 0;
  for (Py_ssize_t i = 0; i < n_arcs; ++i) {
    max_state = std::max(max_state, arcs[4 * i + 0]);
    max_state = std::max(max_state, arcs[4 * i + 1]);
  }
  for (Py_ssize_t i = 0; i < n_finals; ++i)
    max_state = std::max(max_state, finals[i]);

  auto *f = new StdVectorFst();
  for (int s = 0; s <= max_state; ++s) f->AddState();
  f->SetStart(0);

  for (Py_ssize_t i = 0; i < n_arcs; ++i) {
    const int from = arcs[4 * i + 0];
    const int to = arcs[4 * i + 1];
    const int il = arcs[4 * i + 2];
    const int ol = arcs[4 * i + 3];
    const int ilab = (il < 0) ? 0 : il + 1;
    const int olab = (ol < 0) ? 0 : ol + 1;
    if (il >= 0) used_tokens->insert(il);
    f->AddArc(from, StdArc(ilab, olab, Weight::One(), to));
  }
  for (Py_ssize_t i = 0; i < n_finals; ++i)
    f->SetFinal(finals[i], Weight::One());

  return f;
}

// Determinize + minimize in place (functional transducers / acceptors). The
// grammar's number ranges create huge shared-prefix trees; minimization
// collapses them, which is what keeps compose + shortest-path fast at decode.
static void optimize(StdVectorFst *f) {
  try {
    StdVectorFst tmp;
    fst::Determinize(*f, &tmp);
    fst::Minimize(&tmp);
    *f = tmp;
  } catch (const std::exception &) {
    // Leave f unchanged if it is not determinizable.
  }
}

// ----------------------------------------------------------------------------
// Capsule helpers
// ----------------------------------------------------------------------------

static void capsule_destructor(PyObject *capsule) {
  delete static_cast<Grammar *>(PyCapsule_GetPointer(capsule, CAPSULE_NAME));
}

static Grammar *grammar_from_capsule(PyObject *capsule) {
  return static_cast<Grammar *>(PyCapsule_GetPointer(capsule, CAPSULE_NAME));
}

static int get_buffer(PyObject *obj, Py_buffer *view) {
  return PyObject_GetBuffer(obj, view, PyBUF_CONTIG_RO);
}

// ----------------------------------------------------------------------------
// Python entry points
// ----------------------------------------------------------------------------

static PyObject *py_build_grammar(PyObject *, PyObject *args) {
  int num_tokens, blank_id;
  PyObject *arcs_obj, *finals_obj;
  if (!PyArg_ParseTuple(args, "iiOO", &num_tokens, &blank_id, &arcs_obj,
                        &finals_obj))
    return nullptr;

  Py_buffer arcs_view, finals_view;
  if (get_buffer(arcs_obj, &arcs_view) != 0) return nullptr;
  if (get_buffer(finals_obj, &finals_view) != 0) {
    PyBuffer_Release(&arcs_view);
    return nullptr;
  }

  const auto *arcs = static_cast<const int32_t *>(arcs_view.buf);
  const Py_ssize_t n_arcs = arcs_view.len / (4 * sizeof(int32_t));
  const auto *finals = static_cast<const int32_t *>(finals_view.buf);
  const Py_ssize_t n_finals = finals_view.len / sizeof(int32_t);

  Grammar *grammar = new Grammar();
  try {
    std::unordered_set<int> used;
    StdVectorFst *char2sen =
        build_char2sen(arcs, n_arcs, finals, n_finals, &used);

    grammar->tokens.assign(used.begin(), used.end());
    std::sort(grammar->tokens.begin(), grammar->tokens.end());

    StdVectorFst *token2char = build_token2char(grammar->tokens, blank_id + 1);

    // Collapse the expanded grammar tree before composing.
    optimize(char2sen);

    // Compose token2char o char2sen (match token2char olabels to char2sen
    // ilabels): sort the right operand by input label.
    fst::ArcSort(char2sen, fst::ILabelCompare<StdArc>());

    grammar->fst = new StdVectorFst();
    fst::Compose(*token2char, *char2sen, grammar->fst);
    fst::RmEpsilon(grammar->fst);
    // Determinize + minimize so decode-time search stays small, then sort by
    // input label for fast composition against the per-frame lattice.
    optimize(grammar->fst);
    fst::ArcSort(grammar->fst, fst::ILabelCompare<StdArc>());

    delete token2char;
    delete char2sen;
  } catch (const std::exception &e) {
    PyBuffer_Release(&arcs_view);
    PyBuffer_Release(&finals_view);
    delete grammar;
    PyErr_SetString(PyExc_RuntimeError, e.what());
    return nullptr;
  }

  PyBuffer_Release(&arcs_view);
  PyBuffer_Release(&finals_view);

  if (grammar->fst->NumStates() == 0) {
    delete grammar;
    PyErr_SetString(PyExc_RuntimeError, "Grammar composed to an empty FST");
    return nullptr;
  }
  return PyCapsule_New(grammar, CAPSULE_NAME, capsule_destructor);
}

static PyObject *py_save_grammar(PyObject *, PyObject *args) {
  PyObject *capsule;
  const char *path;
  if (!PyArg_ParseTuple(args, "Os", &capsule, &path)) return nullptr;
  Grammar *g = grammar_from_capsule(capsule);
  if (!g) return nullptr;
  if (!g->fst->Write(path)) {
    PyErr_Format(PyExc_IOError, "Failed to write grammar to %s", path);
    return nullptr;
  }
  // Sidecar with the grammar token set (needed for exact decode pruning).
  std::ofstream tok(std::string(path) + ".tokens", std::ios::binary);
  int32_t n = static_cast<int32_t>(g->tokens.size());
  tok.write(reinterpret_cast<const char *>(&n), sizeof(n));
  for (int t : g->tokens) {
    int32_t v = t;
    tok.write(reinterpret_cast<const char *>(&v), sizeof(v));
  }
  Py_RETURN_NONE;
}

static PyObject *py_load_grammar(PyObject *, PyObject *args) {
  const char *path;
  if (!PyArg_ParseTuple(args, "s", &path)) return nullptr;
  StdVectorFst *fst = StdVectorFst::Read(path);
  if (!fst) {
    PyErr_Format(PyExc_IOError, "Failed to read grammar from %s", path);
    return nullptr;
  }
  Grammar *g = new Grammar();
  g->fst = fst;
  std::ifstream tok(std::string(path) + ".tokens", std::ios::binary);
  if (tok) {
    int32_t n = 0;
    tok.read(reinterpret_cast<char *>(&n), sizeof(n));
    for (int32_t i = 0; i < n; ++i) {
      int32_t v = 0;
      tok.read(reinterpret_cast<char *>(&v), sizeof(v));
      g->tokens.push_back(v);
    }
  }
  return PyCapsule_New(g, CAPSULE_NAME, capsule_destructor);
}

// Enumerate every initial->final path of an acyclic FST as (cost, olabels).
struct PathResult {
  double cost;
  std::vector<int> olabels;
};

static void enumerate_paths(const StdVectorFst &fst, StateId state, double acc,
                            std::vector<int> &labels,
                            std::vector<PathResult> &out) {
  Weight final_w = fst.Final(state);
  if (final_w != Weight::Zero())
    out.push_back({acc + final_w.Value(), labels});
  for (fst::ArcIterator<StdVectorFst> ait(fst, state); !ait.Done(); ait.Next()) {
    const StdArc &arc = ait.Value();
    labels.push_back(arc.olabel);
    enumerate_paths(fst, arc.nextstate, acc + arc.weight.Value(), labels, out);
    labels.pop_back();
  }
}

static PyObject *py_decode(PyObject *, PyObject *args) {
  PyObject *capsule, *logits_obj;
  int T, V, blank_id;
  double beam;
  double token_bonus = 0.0;
  if (!PyArg_ParseTuple(args, "OOiiid|d", &capsule, &logits_obj, &T, &V,
                        &blank_id, &beam, &token_bonus))
    return nullptr;

  Grammar *g = grammar_from_capsule(capsule);
  if (!g) return nullptr;

  Py_buffer lv;
  if (get_buffer(logits_obj, &lv) != 0) return nullptr;
  if (lv.len < static_cast<Py_ssize_t>(sizeof(float)) * T * V) {
    PyBuffer_Release(&lv);
    PyErr_SetString(PyExc_ValueError, "logprobs buffer too small for T*V");
    return nullptr;
  }
  const auto *lp = static_cast<const float *>(lv.buf);

  try {
    // Per-frame lattice: an arc only for each grammar token (+ blank). With a
    // positive beam, keep only tokens whose log-prob is within `beam` of the
    // best candidate (grammar tokens or blank) in that frame -- a big speedup
    // for many-frame / character-level grammars. The best candidate always
    // survives, so a path always exists.
    StdVectorFst acc;
    StateId prev = acc.AddState();
    acc.SetStart(prev);
    for (int t = 0; t < T; ++t) {
      const float *row = lp + static_cast<size_t>(t) * V;
      StateId next = acc.AddState();

      float threshold = -std::numeric_limits<float>::infinity();
      if (beam > 0.0) {
        float best = row[blank_id];
        for (int id : g->tokens) best = std::max(best, row[id]);
        threshold = best - static_cast<float>(beam);
      }

      for (int id : g->tokens)
        if (row[id] >= threshold)
          acc.AddArc(prev, StdArc(id + 1, id + 1, Weight(-row[id]), next));
      if (row[blank_id] >= threshold)
        acc.AddArc(prev, StdArc(blank_id + 1, blank_id + 1,
                                Weight(-row[blank_id]), next));
      prev = next;
    }
    acc.SetFinal(prev, Weight::One());

    StdVectorFst composed;
    fst::Compose(acc, *g->fst, &composed);

    std::vector<int> best_tokens;
    double best_cost = 0.0, second_cost = 0.0;
    bool have_best = false, have_second = false;

    if (composed.NumStates() > 0 && composed.Start() != fst::kNoStateId) {
      fst::Project(&composed, fst::ProjectType::OUTPUT);
      // Word-insertion reward: cheapen every emitting arc (olabel > 0) by
      // token_bonus so a longer parse isn't beaten purely by its token count.
      // Each collapsed token maps to exactly one emitting arc, so the total
      // reward on a path is token_bonus * (#emitted tokens) -- added back below
      // to report the true acoustic cost. The lattice is an acyclic per-frame
      // DAG, so the (now possibly negative) weights are safe for ShortestPath.
      if (token_bonus != 0.0) {
        const float bonus = static_cast<float>(token_bonus);
        for (StateId s = 0; s < composed.NumStates(); ++s)
          for (fst::MutableArcIterator<StdVectorFst> ait(&composed, s);
               !ait.Done(); ait.Next()) {
            StdArc arc = ait.Value();
            if (arc.olabel > 0) {
              arc.weight = Weight(arc.weight.Value() - bonus);
              ait.SetValue(arc);
            }
          }
      }
      StdVectorFst nbest;
      fst::ShortestPath(composed, &nbest, /*nshortest=*/2, /*unique=*/true);

      std::vector<PathResult> paths;
      std::vector<int> labels;
      if (nbest.Start() != fst::kNoStateId)
        enumerate_paths(nbest, nbest.Start(), 0.0, labels, paths);
      std::sort(paths.begin(), paths.end(),
                [](const PathResult &a, const PathResult &b) {
                  return a.cost < b.cost;
                });

      if (!paths.empty()) {
        have_best = true;
        int n_emit = 0;
        for (int ol : paths[0].olabels)
          if (ol > 0) {
            best_tokens.push_back(ol - 1);
            ++n_emit;
          }
        // Undo the reward so the caller sees the true acoustic cost.
        best_cost = paths[0].cost + token_bonus * n_emit;
      }
      if (paths.size() > 1) {
        have_second = true;
        int n_emit2 = 0;
        for (int ol : paths[1].olabels)
          if (ol > 0) ++n_emit2;
        second_cost = paths[1].cost + token_bonus * n_emit2;
      }
    }

    PyBuffer_Release(&lv);

    PyObject *tok_list = PyList_New(static_cast<Py_ssize_t>(best_tokens.size()));
    if (!tok_list) return nullptr;
    for (size_t i = 0; i < best_tokens.size(); ++i)
      PyList_SetItem(tok_list, static_cast<Py_ssize_t>(i),
                     PyLong_FromLong(best_tokens[i]));

    PyObject *best_obj =
        have_best ? PyFloat_FromDouble(best_cost) : (Py_INCREF(Py_None), Py_None);
    PyObject *second_obj = have_second ? PyFloat_FromDouble(second_cost)
                                       : (Py_INCREF(Py_None), Py_None);
    return Py_BuildValue("(NNN)", tok_list, best_obj, second_obj);
  } catch (const std::exception &e) {
    PyBuffer_Release(&lv);
    PyErr_SetString(PyExc_RuntimeError, e.what());
    return nullptr;
  }
}

// ----------------------------------------------------------------------------
// Module definition
// ----------------------------------------------------------------------------

static PyMethodDef methods[] = {
    {"build_grammar", py_build_grammar, METH_VARARGS,
     "build_grammar(num_tokens, blank_id, arcs_i32, finals_i32) -> capsule"},
    {"save_grammar", py_save_grammar, METH_VARARGS,
     "save_grammar(capsule, path)"},
    {"load_grammar", py_load_grammar, METH_VARARGS,
     "load_grammar(path) -> capsule"},
    {"decode", py_decode, METH_VARARGS,
     "decode(capsule, logprobs_f32, T, V, blank_id, beam) -> "
     "(token_ids, best_cost, second_cost)"},
    {nullptr, nullptr, 0, nullptr},
};

static struct PyModuleDef moduledef = {
    PyModuleDef_HEAD_INIT, "_fst", "OpenFST grammar build + CTC decode", -1,
    methods, nullptr, nullptr, nullptr, nullptr,
};

PyMODINIT_FUNC PyInit__fst(void) { return PyModule_Create(&moduledef); }
