import unittest

from hass_api import SatelliteInfo, Tool
from tool_mapping import HomeGeometry, map_tool_call


def _tool(required=None, name_map=None, target_schema=None):
    return Tool(
        name="demo",
        tool={
            "type": "function",
            "function": {
                "name": "demo",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "target": target_schema or {"type": "string"},
                        "note": {"type": "string"},
                    },
                    "required": required or [],
                },
            },
        },
        name_map=name_map or {},
    )


class ToolMappingTests(unittest.TestCase):
    def test_omitted_required_field_blocks_script(self):
        call = map_tool_call(_tool(required=["target"]), {})

        self.assertFalse(call.can_run)
        self.assertEqual(["target"], call.missing_required)

    def test_unknown_model_argument_is_not_forwarded(self):
        call = map_tool_call(_tool(), {"note": "ok", "invented": "unsafe"})

        self.assertEqual({"note": "ok"}, call.variables)

    def test_ambiguous_name_uses_satellite_area(self):
        tool = _tool(
            required=["target"],
            name_map={
                "target": {"Speaker": ["media_player.kitchen", "media_player.office"]}
            },
        )
        geometry = HomeGeometry(
            entity_areas={
                "media_player.kitchen": "kitchen",
                "media_player.office": "office",
            }
        )

        call = map_tool_call(
            tool,
            {"target": "Speaker"},
            SatelliteInfo(area_id="office"),
            geometry,
        )

        self.assertTrue(call.can_run)
        self.assertEqual("media_player.office", call.variables["target"])

    def test_multiple_names_resolve_in_order_and_deduplicate_ids(self):
        tool = _tool(
            required=["target"],
            target_schema={
                "type": "array",
                "items": {
                    "type": "string",
                    "enum": ["Desk", "Desk Light", "Kitchen"],
                },
                "uniqueItems": True,
            },
            name_map={
                "target": {
                    "Desk": ["light.desk"],
                    "Desk Light": ["light.desk"],
                    "Kitchen": ["light.kitchen"],
                }
            },
        )

        call = map_tool_call(tool, {"target": ["Desk", "Kitchen", "Desk Light"]})

        self.assertTrue(call.can_run)
        self.assertEqual(["light.desk", "light.kitchen"], call.variables["target"])

    def test_one_unresolved_name_drops_entire_multiple_field(self):
        tool = _tool(
            required=["target"],
            target_schema={
                "type": "array",
                "items": {"type": "string", "enum": ["Desk", "Kitchen"]},
                "uniqueItems": True,
            },
            name_map={
                "target": {
                    "Desk": ["light.desk"],
                    "Kitchen": ["light.kitchen"],
                }
            },
        )

        call = map_tool_call(tool, {"target": ["Desk", "Garage"]})

        self.assertFalse(call.can_run)
        self.assertNotIn("target", call.variables)
        self.assertEqual(["Garage"], call.unresolved["target"])
        self.assertEqual(["target"], call.missing_required)

    def test_one_ambiguous_name_drops_entire_multiple_field(self):
        tool = _tool(
            required=["target"],
            target_schema={
                "type": "array",
                "items": {"type": "string", "enum": ["Desk", "Speaker"]},
                "uniqueItems": True,
            },
            name_map={
                "target": {
                    "Desk": ["light.desk"],
                    "Speaker": [
                        "media_player.kitchen",
                        "media_player.office",
                    ],
                }
            },
        )

        call = map_tool_call(tool, {"target": ["Desk", "Speaker"]})

        self.assertFalse(call.can_run)
        self.assertNotIn("target", call.variables)
        self.assertEqual(
            ["media_player.kitchen", "media_player.office"],
            call.ambiguous["target"],
        )

    def test_scalar_is_rejected_for_multiple_name_field(self):
        tool = _tool(
            required=["target"],
            target_schema={
                "type": "array",
                "items": {"type": "string", "enum": ["Desk"]},
                "uniqueItems": True,
            },
            name_map={"target": {"Desk": ["light.desk"]}},
        )

        call = map_tool_call(tool, {"target": "Desk"})

        self.assertFalse(call.can_run)
        self.assertNotIn("target", call.variables)
        self.assertEqual("Desk", call.unresolved["target"])

    def test_multiple_select_rejects_unknown_option(self):
        tool = _tool(
            required=["target"],
            target_schema={
                "type": "array",
                "items": {"type": "string", "enum": ["email", "sms"]},
                "uniqueItems": True,
            },
        )

        call = map_tool_call(tool, {"target": ["email", "carrier pigeon"]})

        self.assertFalse(call.can_run)
        self.assertNotIn("target", call.variables)
        self.assertEqual(["email", "carrier pigeon"], call.unresolved["target"])

    def test_scalar_select_rejects_unknown_option(self):
        tool = _tool(
            required=["target"],
            target_schema={"type": "string", "enum": ["on", "off"]},
        )

        call = map_tool_call(tool, {"target": "delete"})

        self.assertFalse(call.can_run)
        self.assertNotIn("target", call.variables)
        self.assertEqual("delete", call.unresolved["target"])

    def test_number_rejects_wrong_type_and_out_of_range_value(self):
        tool = _tool(
            required=["target"],
            target_schema={
                "type": "integer",
                "minimum": 0,
                "maximum": 100,
            },
        )

        for value in ("50", -1, 101):
            with self.subTest(value=value):
                call = map_tool_call(tool, {"target": value})
                self.assertFalse(call.can_run)
                self.assertNotIn("target", call.variables)
                self.assertEqual(value, call.unresolved["target"])

    def test_number_enforces_step(self):
        tool = _tool(
            required=["target"],
            target_schema={"type": "number", "multipleOf": 0.5},
        )

        self.assertTrue(map_tool_call(tool, {"target": 1.5}).can_run)
        self.assertFalse(map_tool_call(tool, {"target": 1.2}).can_run)

    def test_array_enforces_item_bounds(self):
        tool = _tool(
            required=["target"],
            target_schema={
                "type": "array",
                "items": {"type": "integer", "minimum": 0, "maximum": 255},
                "minItems": 3,
                "maxItems": 3,
            },
        )

        self.assertTrue(map_tool_call(tool, {"target": [0, 127, 255]}).can_run)
        for value in ([0, 255], [0, 127, 256], [-1, 127, 255]):
            with self.subTest(value=value):
                call = map_tool_call(tool, {"target": value})
                self.assertFalse(call.can_run)
                self.assertNotIn("target", call.variables)
                self.assertEqual(value, call.unresolved["target"])


if __name__ == "__main__":
    unittest.main()
