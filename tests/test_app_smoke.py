"""Smoke test of app.py page logic with a *stubbed* Streamlit (and plotly if absent).

This executes every page function against the real trained model and data, so
Python-level errors (bad names, column mismatches, data-shape bugs) are caught.
It does NOT render a browser UI - run `streamlit run app.py` for that.
"""
import importlib
import io
import sys
import types
import unittest
from unittest import mock

from config import settings


class Col(mock.MagicMock):
    """Stub for st.columns()/st.tabs() children: widgets return realistic values."""

    def selectbox(self, label, options, index=0, **k):
        return options[index]

    def number_input(self, label, **k):
        return k.get("value", 0)

    def multiselect(self, label, options, default=None, **k):
        return default if default is not None else options


def make_streamlit_stub(page="Dashboard", upload=None, submit=True):
    st = types.ModuleType("streamlit")
    M = mock.MagicMock

    def passthrough(*a, **k):
        if len(a) == 1 and callable(a[0]) and not k:
            a[0].clear = lambda: None
            return a[0]
        def deco(fn):
            fn.clear = lambda: None
            return fn
        return deco

    st.cache_data = passthrough
    st.cache_resource = passthrough
    st.cache_data.clear = lambda: None
    for n in ("set_page_config", "markdown", "title", "header", "subheader", "caption",
              "write", "warning", "error", "info", "success", "dataframe", "plotly_chart",
              "image", "text", "code", "divider", "rerun"):
        setattr(st, n, M(name=n))
    st.metric = M()
    st.spinner = lambda *a, **k: M()
    st.form = lambda *a, **k: M()
    st.form_submit_button = lambda *a, **k: submit
    st.button = lambda *a, **k: False
    st.download_button = M()
    st.columns = lambda spec, **k: [Col() for _ in range(spec if isinstance(spec, int) else len(spec))]
    st.tabs = lambda labels: [Col() for _ in labels]
    st.selectbox = lambda label, options, index=0, **k: options[index]
    st.multiselect = lambda label, options, default=None, **k: default if default is not None else options
    st.number_input = lambda label, **k: k.get("value", 0)
    st.file_uploader = lambda *a, **k: upload
    sb = M()
    sb.radio = lambda label, options, **k: page
    st.sidebar = sb
    return st


def load_app(**kw):
    """Import a fresh copy of app.py bound to a stubbed streamlit (and plotly if absent)."""
    import pandas, numpy  # noqa: F401  (make sure heavy deps are imported first)
    stubs = {"streamlit": make_streamlit_stub(**kw)}
    try:
        import plotly  # noqa
    except ImportError:
        px, go = mock.MagicMock(), mock.MagicMock()
        pl = types.ModuleType("plotly"); pl.express, pl.graph_objects = px, go
        stubs.update({"plotly": pl, "plotly.express": px, "plotly.graph_objects": go})
    saved = {k: sys.modules.get(k) for k in list(stubs) + ["app"]}
    try:
        sys.modules.update(stubs)
        sys.modules.pop("app", None)
        app = importlib.import_module("app")
    finally:
        for k, v in saved.items():
            if v is None:
                sys.modules.pop(k, None)
            else:
                sys.modules[k] = v
    return app


ready = all(p.exists() for p in (settings.MODEL_PATH, settings.METADATA_PATH,
                                 settings.CLEAN_DATA_PATH, settings.BACKGROUND_PATH,
                                 settings.CURVES_PATH))


@unittest.skipUnless(ready, "train first: python -m src.train")
class TestAppPages(unittest.TestCase):
    def run_page(self, name, **kw):
        app = load_app(page=name, **kw)
        app.main()
        return app

    def test_every_page_renders_without_exception(self):
        for page in ["Dashboard", "Customer Prediction", "Analytics",
                     "Model Performance", "Batch Prediction"]:
            with self.subTest(page=page):
                self.run_page(page)

    def test_prediction_page_calls_charts_and_actions(self):
        app = self.run_page("Customer Prediction")
        self.assertTrue(app.st.plotly_chart.called)
        lines = [c.args[0] for c in app.st.write.call_args_list if c.args]
        bullets = [l for l in lines if isinstance(l, str) and l.startswith("- ")]
        self.assertGreaterEqual(len(bullets), 4)   # explanations + retention actions
        self.assertFalse(app.st.error.called)

    def test_batch_page_with_valid_upload(self):
        import pandas as pd
        csv = pd.read_csv(settings.CLEAN_DATA_PATH).drop(columns=["Churn"]).head(25)
        app = self.run_page("Batch Prediction", upload=io.StringIO(csv.to_csv(index=False)))
        self.assertTrue(app.st.download_button.call_count >= 2)  # template + predictions

    def test_batch_page_with_invalid_upload_shows_error(self):
        app = self.run_page("Batch Prediction", upload=io.StringIO("a,b\n1,2\n"))
        self.assertTrue(app.st.error.called)

    def test_setup_page_when_model_missing(self):
        app = load_app()
        with mock.patch.object(app, "artifacts_ready", return_value=False):
            app.main()
        self.assertTrue(app.st.error.called)


if __name__ == "__main__":
    unittest.main()
