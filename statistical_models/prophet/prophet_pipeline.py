"""
Prophet Per-Series Pipeline - Walmart Store Sales Forecasting
----------------------------------------------------------------
XGBoost-ის sklearn Pipeline-ის მსგავსი "ერთი ობიექტი", რომელიც
3254 ცალკეულ Prophet მოდელს (თითო Store-Dept წყვილზე) ინახავს
ერთ picklable/loadable ერთეულად.

გამოყენება:
    forecaster = ProphetPerSeriesForecaster(
        holidays_df=holidays_df_narrow,
        changepoint_prior_scale=0.01,
        seasonality_mode="additive",
    )
    forecaster.fit(train_df)                      # ვწვრთნით ყველა სერიას
    preds = forecaster.predict(test_df)            # ვაკეთებთ prediction-ს

MLflow-ზე შენახვა/ჩატვირთვა:
    log_forecaster(forecaster, artifact_path="model")
    loaded = load_forecaster(model_uri)             # inference-ისთვის
"""

import numpy as np
import pandas as pd
from prophet import Prophet
import mlflow
import mlflow.pyfunc


REGRESSORS = ["Temperature", "Fuel_Price", "CPI", "Unemployment", "MarkDown_total"]


class ProphetPerSeriesForecaster:
    """თითო (Store, Dept) წყვილზე ცალკე Prophet მოდელი. sklearn-style fit/predict."""

    def __init__(self, holidays_df=None, changepoint_prior_scale=0.01,
                 seasonality_mode="additive", holidays_prior_scale=10.0,
                 use_regressors=False, regressor_list=None, min_obs=10):
        self.holidays_df = holidays_df
        self.changepoint_prior_scale = changepoint_prior_scale
        self.seasonality_mode = seasonality_mode
        self.holidays_prior_scale = holidays_prior_scale
        self.use_regressors = use_regressors
        self.regressor_list = regressor_list or REGRESSORS
        self.min_obs = min_obs

        self.models_ = {}       # (store, dept) -> fitted Prophet model
        self.fallbacks_ = {}    # (store, dept) -> fallback mean, თუ სერია ძალიან მოკლეა

    def fit(self, train_df: pd.DataFrame, verbose=True):
        keys = train_df[["Store", "Dept"]].drop_duplicates().itertuples(index=False)
        keys = list(keys)

        iterator = keys
        if verbose:
            from tqdm import tqdm
            iterator = tqdm(keys, desc="Fitting Prophet per Store-Dept")

        for store, dept in iterator:
            g = train_df[(train_df.Store == store) & (train_df.Dept == dept)]
            df = g.rename(columns={"Date": "ds", "Weekly_Sales": "y"})
            available_regs = [r for r in self.regressor_list if r in df.columns] if self.use_regressors else []
            df = df[["ds", "y"] + available_regs].sort_values("ds")

            if len(df) < self.min_obs:
                self.fallbacks_[(store, dept)] = g["Weekly_Sales"].mean() if len(g) else 0.0
                continue

            m = Prophet(
                holidays=self.holidays_df,
                changepoint_prior_scale=self.changepoint_prior_scale,
                seasonality_mode=self.seasonality_mode,
                holidays_prior_scale=self.holidays_prior_scale,
            )
            for reg in available_regs:
                m.add_regressor(reg)
            m.fit(df)
            self.models_[(store, dept)] = m

        return self

    def predict(self, future_df: pd.DataFrame) -> pd.DataFrame:
        """
        future_df: Store, Dept, Date სვეტები (+ regressor სვეტები, თუ use_regressors=True)
        აბრუნებს: Store, Dept, Date, Weekly_Sales_Pred
        """
        results = []
        keys = future_df[["Store", "Dept"]].drop_duplicates().itertuples(index=False)

        for store, dept in keys:
            g = future_df[(future_df.Store == store) & (future_df.Dept == dept)]

            if (store, dept) in self.models_:
                m = self.models_[(store, dept)]
                available_regs = [r for r in self.regressor_list if r in g.columns] if self.use_regressors else []
                fut = g.rename(columns={"Date": "ds"})[["ds"] + available_regs].sort_values("ds")
                forecast = m.predict(fut)
                out = g[["Store", "Dept", "Date"]].copy()
                out["Weekly_Sales_Pred"] = np.clip(forecast["yhat"].values, a_min=0, a_max=None)
            else:
                # ან fallback (მოკლე სერია), ან სულ ახალი, ტრენინგში არნახული (Store, Dept)
                fallback = self.fallbacks_.get((store, dept), 0.0)
                out = g[["Store", "Dept", "Date"]].copy()
                out["Weekly_Sales_Pred"] = fallback

            results.append(out)

        return pd.concat(results, ignore_index=True)


# ---------------------------------------------------------------------------
# MLflow pyfunc wrapper - შენახვა/ჩატვირთვა
# ---------------------------------------------------------------------------

class _ProphetPyfuncWrapper(mlflow.pyfunc.PythonModel):
    """მცირე wrapper, რომ ProphetPerSeriesForecaster mlflow.pyfunc-ს მიესადაგოს."""

    def load_context(self, context):
        import cloudpickle
        with open(context.artifacts["forecaster"], "rb") as f:
            self.forecaster = cloudpickle.load(f)

    def predict(self, context, model_input: pd.DataFrame) -> pd.DataFrame:
        return self.forecaster.predict(model_input)


def log_forecaster(forecaster: ProphetPerSeriesForecaster, artifact_path: str = "model",
                    registered_model_name: str = None):
    """
    ინახავს მთელ forecaster-ს (ყველა 3254 fitted Prophet მოდელს) ერთ MLflow model artifact-ად,
    cloudpickle serialization-ით (skops-ის თავიდან ასაცილებლად, radgan Prophet custom class-ია).
    """
    import cloudpickle
    import tempfile
    import os

    with tempfile.TemporaryDirectory() as tmp_dir:
        pkl_path = os.path.join(tmp_dir, "forecaster.pkl")
        with open(pkl_path, "wb") as f:
            cloudpickle.dump(forecaster, f)

        mlflow.pyfunc.log_model(
            artifact_path=artifact_path,
            python_model=_ProphetPyfuncWrapper(),
            artifacts={"forecaster": pkl_path},
            registered_model_name=registered_model_name,
        )
    print(f"Forecaster დალოგინდა MLflow-ზე, artifact_path='{artifact_path}'")


def load_forecaster(model_uri: str):
    """
    ჩატვირთვა inference-ისთვის, მაგ:
        model_uri = f"runs:/{run_id}/model"
        # ან: model_uri = "models:/ProphetWalmartForecaster/Production"
    """
    loaded_model = mlflow.pyfunc.load_model(model_uri)
    return loaded_model  # loaded_model.predict(future_df) გამოსაძახებლად მზადაა
