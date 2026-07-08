import ast
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def parse_module(relative_path):
    return ast.parse((ROOT / relative_path).read_text())


def find_function(module, name):
    for node in module.body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise AssertionError(f"Function not found: {name}")


def names_in(node):
    return {child.id for child in ast.walk(node) if isinstance(child, ast.Name)}


def calls_to(node, name):
    return [
        child
        for child in ast.walk(node)
        if isinstance(child, ast.Call)
        and isinstance(child.func, ast.Name)
        and child.func.id == name
    ]


class ForceExpandValidationBaselineTests(unittest.TestCase):
    def test_config_declares_force_expand_validation_baseline_as_boolean(self):
        config = parse_module("config.py")

        for node in config.body:
            if not isinstance(node, ast.Assign):
                continue
            if any(
                isinstance(target, ast.Name)
                and target.id == "FORCE_EXPAND_VALIDATION_BASELINE"
                for target in node.targets
            ):
                self.assertIsInstance(node.value, ast.Constant)
                self.assertIsInstance(node.value.value, bool)
                return

        self.fail("FORCE_EXPAND_VALIDATION_BASELINE is not defined in config.py")

    def test_pipeline_passes_force_expand_validation_baseline_to_training(self):
        pipeline = parse_module("pipeline.py")
        train_model = find_function(pipeline, "train_model")

        ffno_train_calls = calls_to(train_model, "ffno_train_model")
        self.assertEqual(len(ffno_train_calls), 1)

        keywords = {
            keyword.arg: keyword.value
            for keyword in ffno_train_calls[0].keywords
        }
        self.assertIn("force_expand_validation_baseline", keywords)
        self.assertIsInstance(
            keywords["force_expand_validation_baseline"],
            ast.Name,
        )
        self.assertEqual(
            keywords["force_expand_validation_baseline"].id,
            "FORCE_EXPAND_VALIDATION_BASELINE",
        )

    def test_ffno_train_model_supports_forced_expand_validation_baseline(self):
        ffnonet = parse_module("FFNONet.py")
        train_fn = find_function(ffnonet, "ffno_train_model")

        arg_names = [arg.arg for arg in train_fn.args.args]
        arg_names += [arg.arg for arg in train_fn.args.kwonlyargs]
        self.assertIn("force_expand_validation_baseline", arg_names)

        kw_defaults_by_arg = dict(
            zip(
                [arg.arg for arg in train_fn.args.kwonlyargs],
                train_fn.args.kw_defaults,
            )
        )
        default = kw_defaults_by_arg["force_expand_validation_baseline"]
        self.assertIsInstance(default, ast.Constant)
        self.assertIs(default.value, False)

        validation_guards = [
            node
            for node in ast.walk(train_fn)
            if isinstance(node, ast.If)
            and {"expand_from_checkpoint", "force_expand_validation_baseline"}
            <= names_in(node.test)
        ]
        self.assertTrue(
            validation_guards,
            "forced flag is not part of the expand baseline guard",
        )

        save_calls = calls_to(train_fn, "save_checkpoint_fsdp")
        epoch_zero_saves = [
            call
            for call in save_calls
            if any(
                keyword.arg == "epoch"
                and isinstance(keyword.value, ast.Constant)
                and keyword.value.value == 0
                for keyword in call.keywords
            )
        ]
        self.assertTrue(
            epoch_zero_saves,
            "expand baseline validation is not saved as an epoch-0 best checkpoint",
        )

        train_calls = calls_to(train_fn, "train")
        self.assertEqual(len(train_calls), 1)
        train_keywords = {
            keyword.arg: keyword.value
            for keyword in train_calls[0].keywords
        }
        self.assertIn("best_val_init", train_keywords)
        self.assertIsInstance(train_keywords["best_val_init"], ast.Name)
        self.assertEqual(
            train_keywords["best_val_init"].id,
            "effective_best_val_init",
        )


if __name__ == "__main__":
    unittest.main()
