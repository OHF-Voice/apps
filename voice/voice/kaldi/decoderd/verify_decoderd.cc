// Resident grammar decoder for wake verification / alignment.
//
// Loads the acoustic model, i-vector extractor, and ALL of the verifier's
// decoding graphs once, then decodes wav files on demand: one wav path per
// stdin line, one result line per graph on stdout.
//
// The point of handling every graph in one process is that the acoustic model
// is then evaluated ONCE per utterance instead of once per graph. The four
// verifier graphs score the same audio against different grammars, so the
// nnet3 forward pass -- about 95 ms of a 1.4 s window on a Pi 4 -- was being
// done four times over identical input. Here a single feature pipeline feeds a
// single decodable, and the graph searches share it.
//
//   verify_decoderd [options] <final.mdl> <tag> <HCLG.fst> <word_boundary.int|-> ...
//
// with the last three arguments repeated once per graph.
//
// Protocol (line oriented):
//   stdin:  <wav-path>
//   stdout: exactly one line per graph, in the order the graphs were given:
//             OK <tag> <total_best_path_cost> <n_output_frames> [wordid:beg:end ...]
//             ERR <tag> <message>
//   "READY" is printed once after loading. Word times are in OUTPUT frames
//   (multiply by 0.01 * frame-subsampling-factor for seconds); word ids map
//   through the graph's words.txt. Word alignment entries are only produced
//   for graphs given a word_boundary.int file.
//
// Two things make the sharing exact rather than approximate:
//
//   * The decoders are advanced ONE output frame at a time, in lockstep. The
//     looped decodable computes log-posteriors in chunks and refuses frames
//     requested out of order ("Frames must be accessed in order"), so letting
//     any one decoder run ahead would leave the others asking for frames the
//     cache has already passed.
//
//   * Each graph keeps its OWN decoder config. They share the acoustic model,
//     not the search: --max-active must not apply to the confusable graph,
//     whose whole job is to find a rival hypothesis that capping prunes away.
//     --uncapped-graphs names the exceptions. (Getting this wrong is not
//     subtle in its effect but is silent in its symptoms: it halves the
//     confusable margin and accepts false positives.)
//
//   * i-vector silence weighting needs a decoder traceback, and there is now
//     one feature stream rather than four, so --silence-weighting-graph picks
//     which graph drives it (default: the first). With the shipped model this
//     turns out not to matter -- the config carries no silence weighting, so
//     the choice is inert -- but it is there if a model enables it.
//
// Costs/alignments are otherwise equivalent to the subprocess pipeline
// (online2-wav-nnet3-latgen-faster | lattice-1best | lattice-align-words |
// lattice-best-path / nbest-to-ctm) with the same options.

#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include "feat/wave-reader.h"
#include "fstext/kaldi-fst-io.h"
#include "lat/determinize-lattice-pruned.h"
#include "lat/lattice-functions.h"
#include "lat/word-align-lattice.h"
#include "nnet3/decodable-online-looped.h"
#include "nnet3/nnet-utils.h"
#include "online2/online-endpoint.h"
#include "online2/online-nnet2-feature-pipeline.h"
#include "online2/online-nnet3-decoding.h"

namespace {

using namespace kaldi;

struct Graph {
  std::string tag;
  fst::Fst<fst::StdArc> *fst = nullptr;
  WordBoundaryInfo *word_boundary = nullptr;
  // Per-graph, because max_active must differ: see --uncapped-graphs.
  LatticeFasterDecoderConfig opts;
  std::unique_ptr<LatticeFasterOnlineDecoder> decoder;
};

// Split a comma-separated option value.
std::vector<std::string> SplitCommas(const std::string &value) {
  std::vector<std::string> out;
  std::string item;
  std::istringstream stream(value);
  while (std::getline(stream, item, ',')) {
    if (!item.empty()) {
      out.push_back(item);
    }
  }
  return out;
}

// Total graph+acoustic cost and frame count of a linear best path, matching
// what lattice-best-path reports.
void BestPathCost(const CompactLattice &best_path, double *total_cost,
                  int64 *n_frames) {
  *total_cost = 0.0;
  *n_frames = 0;
  typedef CompactLatticeArc Arc;
  Arc::StateId state = best_path.Start();
  while (state != fst::kNoStateId) {
    const auto final_weight = best_path.Final(state);
    if (final_weight != CompactLatticeWeight::Zero()) {
      *total_cost +=
          final_weight.Weight().Value1() + final_weight.Weight().Value2();
      *n_frames += final_weight.String().size();
      break;
    }
    fst::ArcIterator<CompactLattice> aiter(best_path, state);
    if (aiter.Done()) {
      break;
    }
    const Arc &arc = aiter.Value();
    *total_cost += arc.weight.Weight().Value1() + arc.weight.Weight().Value2();
    *n_frames += arc.weight.String().size();
    state = arc.nextstate;
  }
}

}  // namespace

int main(int argc, char *argv[]) {
  using namespace kaldi;
  try {
    const char *usage =
        "Resident wake-verification decoder; see header comment.\n"
        "Usage: verify_decoderd [options] <nnet3-in> "
        "(<tag> <fst-in> <word-boundary-int|->)+\n";

    ParseOptions po(usage);
    OnlineNnet2FeaturePipelineConfig feature_opts;
    nnet3::NnetSimpleLoopedComputationOptions decodable_opts;
    LatticeFasterDecoderConfig decoder_opts;
    WordBoundaryInfoNewOpts word_boundary_opts;
    // unused, but online.conf carries --endpoint.* options and the shared
    // config file must parse
    OnlineEndpointConfig endpoint_opts;

    // Match online2-wav-nnet3-latgen-faster --online=true exactly: chunked
    // feeding + i-vector silence weighting. Skipping this shifts best-path
    // costs by several nats (measured), which is material against the
    // verifier's calibrated thresholds.
    BaseFloat chunk_length_secs = 0.18;
    std::string silence_weighting_graph;
    std::string uncapped_graphs;

    feature_opts.Register(&po);
    decodable_opts.Register(&po);
    decoder_opts.Register(&po);
    endpoint_opts.Register(&po);
    po.Register("chunk-length", &chunk_length_secs,
                "Length of audio chunks fed to the feature pipeline "
                "(matches the reference binary's default)");
    po.Register("silence-weighting-graph", &silence_weighting_graph,
                "Tag of the graph whose traceback drives i-vector silence "
                "weighting (default: the first graph given)");
    po.Register("uncapped-graphs", &uncapped_graphs,
                "Comma-separated graph tags that ignore --max-active. The "
                "confusable graph must be listed: capping the number of "
                "active states prunes the rival hypothesis it exists to find, "
                "which silently accepts false positives");
    po.Read(argc, argv);

    if (po.NumArgs() < 4 || (po.NumArgs() - 1) % 3 != 0) {
      po.PrintUsage();
      return 1;
    }

    const std::string nnet3_rxfilename = po.GetArg(1);
    const int32 num_graphs = (po.NumArgs() - 1) / 3;

    OnlineNnet2FeaturePipelineInfo feature_info(feature_opts);

    TransitionModel trans_model;
    nnet3::AmNnetSimple am_nnet;
    {
      bool binary;
      Input ki(nnet3_rxfilename, &binary);
      trans_model.Read(ki.Stream(), binary);
      am_nnet.Read(ki.Stream(), binary);
      SetBatchnormTestMode(true, &(am_nnet.GetNnet()));
      SetDropoutTestMode(true, &(am_nnet.GetNnet()));
      nnet3::CollapseModel(nnet3::CollapseModelConfig(), &(am_nnet.GetNnet()));
    }
    nnet3::DecodableNnetSimpleLoopedInfo decodable_info(decodable_opts,
                                                        &am_nnet);

    const std::vector<std::string> uncapped = SplitCommas(uncapped_graphs);

    std::vector<Graph> graphs(num_graphs);
    int32 silence_weighting_index = 0;
    for (int32 g = 0; g < num_graphs; g++) {
      graphs[g].tag = po.GetArg(2 + 3 * g);
      graphs[g].opts = decoder_opts;
      if (std::find(uncapped.begin(), uncapped.end(), graphs[g].tag) !=
          uncapped.end()) {
        graphs[g].opts.max_active = std::numeric_limits<int32>::max();
      }
      graphs[g].fst = fst::ReadFstKaldiGeneric(po.GetArg(3 + 3 * g));
      const std::string wb = po.GetArg(4 + 3 * g);
      if (wb != "-") {
        graphs[g].word_boundary =
            new WordBoundaryInfo(word_boundary_opts, wb);
      }
      if (!silence_weighting_graph.empty() &&
          graphs[g].tag == silence_weighting_graph) {
        silence_weighting_index = g;
      }
    }
    if (!silence_weighting_graph.empty() &&
        graphs[silence_weighting_index].tag != silence_weighting_graph) {
      KALDI_ERR << "--silence-weighting-graph=" << silence_weighting_graph
                << " does not name any of the graphs given";
    }

    std::cout << "READY" << std::endl;

    std::string wav_path;
    while (std::getline(std::cin, wav_path)) {
      if (wav_path.empty()) {
        continue;
      }

      std::vector<std::string> lines(num_graphs);
      try {
        // NOTE: MFCC dither draws from the process-global rand() stream, so
        // consecutive requests see different dither than fresh subprocesses
        // would. Measured effect: <= ~0.45 total best-path cost (~0.02 lr
        // units) -- an order of magnitude below the verifier's decision
        // gaps, and the calibration data carries the same dither noise.
        WaveData wave_data;
        {
          Input ki(wav_path);
          wave_data.Read(ki.Stream());
        }
        SubVector<BaseFloat> data(wave_data.Data(), 0);
        const BaseFloat samp_freq = wave_data.SampFreq();

        OnlineNnet2FeaturePipeline feature_pipeline(feature_info);
        // ONE decodable: the acoustic model runs once for all the graphs.
        nnet3::DecodableAmNnetLoopedOnline decodable(
            trans_model, decodable_info, feature_pipeline.InputFeature(),
            feature_pipeline.IvectorFeature());

        for (int32 g = 0; g < num_graphs; g++) {
          // (fst, config): borrows the FST. The (config, FST*) overload
          // takes ownership, which would delete a graph we reuse every turn.
          graphs[g].decoder.reset(
              new LatticeFasterOnlineDecoder(*graphs[g].fst, graphs[g].opts));
          graphs[g].decoder->InitDecoding();
        }

        OnlineSilenceWeighting silence_weighting(
            trans_model, feature_info.silence_weighting_config,
            decodable_opts.frame_subsampling_factor);
        std::vector<std::pair<int32, BaseFloat>> delta_weights;

        int32 chunk_length = static_cast<int32>(samp_freq * chunk_length_secs);
        if (chunk_length <= 0) {
          chunk_length = data.Dim();
        }
        int32 samp_offset = 0;
        while (samp_offset < data.Dim()) {
          int32 num_samp =
              std::min<int32>(chunk_length, data.Dim() - samp_offset);
          SubVector<BaseFloat> wave_part(data, samp_offset, num_samp);
          feature_pipeline.AcceptWaveform(samp_freq, wave_part);
          samp_offset += num_samp;
          if (samp_offset == data.Dim()) {
            feature_pipeline.InputFinished();
          }
          if (silence_weighting.Active() &&
              feature_pipeline.IvectorFeature() != nullptr) {
            silence_weighting.ComputeCurrentTraceback(
                *graphs[silence_weighting_index].decoder);
            silence_weighting.GetDeltaWeights(
                feature_pipeline.NumFramesReady(), &delta_weights);
            feature_pipeline.IvectorFeature()->UpdateFrameWeights(
                delta_weights);
          }
          // Lockstep, one output frame at a time: the looped decodable caches
          // a chunk of log-posteriors and rejects frames requested before its
          // current offset, so no decoder may run ahead of the others.
          while (graphs[0].decoder->NumFramesDecoded() <
                 decodable.NumFramesReady()) {
            for (int32 g = 0; g < num_graphs; g++) {
              graphs[g].decoder->AdvanceDecoding(&decodable, 1);
            }
          }
        }

        for (int32 g = 0; g < num_graphs; g++) {
          Graph &graph = graphs[g];
          graph.decoder->FinalizeDecoding();

          if (graph.decoder->NumFramesDecoded() == 0) {
            lines[g] = "ERR " + graph.tag + " no frames decoded";
            continue;
          }

          Lattice raw_lat;
          graph.decoder->GetRawLattice(&raw_lat, true /* end_of_utterance */);
          CompactLattice clat;
          DeterminizeLatticePhonePrunedWrapper(trans_model, &raw_lat,
                                               graph.opts.lattice_beam, &clat,
                                               graph.opts.det_opts);
          if (clat.NumStates() == 0) {
            lines[g] = "ERR " + graph.tag + " empty lattice";
            continue;
          }

          CompactLattice best_path;
          CompactLatticeShortestPath(clat, &best_path);
          if (best_path.NumStates() == 0) {
            lines[g] = "ERR " + graph.tag + " empty best path";
            continue;
          }

          double total_cost = 0.0;
          int64 n_frames = 0;
          BestPathCost(best_path, &total_cost, &n_frames);

          std::ostringstream out;
          out << "OK " << graph.tag << " " << total_cost << " " << n_frames;

          if (graph.word_boundary != nullptr) {
            CompactLattice aligned;
            if (WordAlignLattice(best_path, trans_model, *graph.word_boundary,
                                 0 /* max_states: no limit */, &aligned)) {
              std::vector<int32> words, times, lengths;
              if (CompactLatticeToWordAlignment(aligned, &words, &times,
                                                &lengths)) {
                for (size_t i = 0; i < words.size(); i++) {
                  if (words[i] == 0) {
                    continue;  // <eps> / silence gap
                  }
                  out << " " << words[i] << ":" << times[i] << ":"
                      << (times[i] + lengths[i]);
                }
              }
            }
          }
          lines[g] = out.str();
        }
      } catch (const std::exception &e) {
        for (int32 g = 0; g < num_graphs; g++) {
          if (lines[g].empty()) {
            lines[g] = "ERR " + graphs[g].tag + " " + e.what();
          }
        }
      }

      // Always exactly one line per graph, so the client can read a fixed
      // number of lines without a terminator.
      for (int32 g = 0; g < num_graphs; g++) {
        if (lines[g].empty()) {
          lines[g] = "ERR " + graphs[g].tag + " no result";
        }
        std::cout << lines[g] << "\n";
      }
      std::cout << std::flush;
    }

    for (int32 g = 0; g < num_graphs; g++) {
      delete graphs[g].word_boundary;
      delete graphs[g].fst;
    }
    return 0;
  } catch (const std::exception &e) {
    std::cerr << e.what() << std::endl;
    return 1;
  }
}
