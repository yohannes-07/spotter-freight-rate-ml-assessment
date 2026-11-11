"""Feature construction for freight-rate models."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


RAW_NUMERIC_COLUMNS = [
    "pickup_lat",
    "pickup_lon",
    "delivery_lat",
    "delivery_lon",
    "distance",
    "weight",
    "market_index",
    "quote_signal",
]
CATEGORICAL_COLUMNS = ["pickup", "delivery", "equipment", "lane"]
EARTH_RADIUS_MILES = 3_958.8


@dataclass
class FreightPreprocessor:
    """Stateful preprocessing fitted on training data only."""

    numeric_medians_: dict[str, float] = field(default_factory=dict, init=False)
    pickup_coordinates_: dict[str, tuple[float, float]] = field(default_factory=dict, init=False)
    delivery_coordinates_: dict[str, tuple[float, float]] = field(default_factory=dict, init=False)
    feature_columns_: list[str] = field(default_factory=list, init=False)

    @staticmethod
    def _coordinate_map(
        frame: pd.DataFrame, city_column: str, lat_column: str, lon_column: str
    ) -> dict[str, tuple[float, float]]:
        coordinates = frame[[city_column, lat_column, lon_column]].copy()
        coordinates[lat_column] = pd.to_numeric(coordinates[lat_column], errors="coerce")
        coordinates[lon_column] = pd.to_numeric(coordinates[lon_column], errors="coerce")
        coordinates = coordinates.dropna()
        grouped = coordinates.groupby(city_column, observed=True)[[lat_column, lon_column]].median()
        return {str(city): (float(row[lat_column]), float(row[lon_column])) for city, row in grouped.iterrows()}

    def fit(self, frame: pd.DataFrame) -> "FreightPreprocessor":
        data = frame.copy()
        self.pickup_coordinates_ = self._coordinate_map(
            data, "pickup", "pickup_lat", "pickup_lon"
        )
        self.delivery_coordinates_ = self._coordinate_map(
            data, "delivery", "delivery_lat", "delivery_lon"
        )

        for column in RAW_NUMERIC_COLUMNS:
            values = pd.to_numeric(data.get(column), errors="coerce")
            if column == "weight":
                values = values.abs()
            median = values.median()
            self.numeric_medians_[column] = float(median) if pd.notna(median) else 0.0

        transformed = self.transform(data, align=False)
        self.feature_columns_ = transformed.columns.tolist()
        return self

    def _fill_coordinates(self, data: pd.DataFrame) -> None:
        pairs = [
            ("pickup", "pickup_lat", "pickup_lon", self.pickup_coordinates_),
            ("delivery", "delivery_lat", "delivery_lon", self.delivery_coordinates_),
        ]
        for city_column, lat_column, lon_column, lookup in pairs:
            city = data[city_column].astype("string")
            fallback_lat = city.map({key: value[0] for key, value in lookup.items()})
            fallback_lon = city.map({key: value[1] for key, value in lookup.items()})
            data[lat_column] = pd.to_numeric(data[lat_column], errors="coerce").fillna(fallback_lat)
            data[lon_column] = pd.to_numeric(data[lon_column], errors="coerce").fillna(fallback_lon)

    @staticmethod
    def _haversine(data: pd.DataFrame) -> np.ndarray:
        lat1 = np.radians(data["pickup_lat"].to_numpy(dtype=float))
        lon1 = np.radians(data["pickup_lon"].to_numpy(dtype=float))
        lat2 = np.radians(data["delivery_lat"].to_numpy(dtype=float))
        lon2 = np.radians(data["delivery_lon"].to_numpy(dtype=float))
        delta_lat = lat2 - lat1
        delta_lon = lon2 - lon1
        value = np.sin(delta_lat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(delta_lon / 2.0) ** 2
        return 2.0 * EARTH_RADIUS_MILES * np.arcsin(np.sqrt(np.clip(value, 0.0, 1.0)))

    def transform(self, frame: pd.DataFrame, *, align: bool = True) -> pd.DataFrame:
        if not self.numeric_medians_:
            raise RuntimeError("FreightPreprocessor must be fitted before transform")

        data = frame.copy()
        for column in ["pickup", "delivery", "equipment"]:
            if column not in data:
                data[column] = "Unknown"
            data[column] = data[column].astype("string").fillna("Unknown")

        for column in RAW_NUMERIC_COLUMNS:
            if column not in data:
                data[column] = np.nan

        weight = pd.to_numeric(data["weight"], errors="coerce")
        data["weight_was_missing"] = weight.isna().astype("int8")
        data["weight_sign_corrected"] = weight.lt(0).astype("int8")
        data["market_index_was_missing"] = pd.to_numeric(
            data["market_index"], errors="coerce"
        ).isna().astype("int8")
        data["weight"] = weight.abs()

        self._fill_coordinates(data)
        for column in RAW_NUMERIC_COLUMNS:
            data[column] = pd.to_numeric(data[column], errors="coerce").fillna(
                self.numeric_medians_[column]
            )

        date = pd.to_datetime(data.get("date"), errors="coerce")
        if date.isna().any():
            raise ValueError(f"Found {int(date.isna().sum())} invalid date values")

        features = data[RAW_NUMERIC_COLUMNS].copy()
        features["weight_was_missing"] = data["weight_was_missing"]
        features["weight_sign_corrected"] = data["weight_sign_corrected"]
        features["market_index_was_missing"] = data["market_index_was_missing"]
        features["lane"] = data["pickup"] + " -> " + data["delivery"]
        features["pickup"] = data["pickup"]
        features["delivery"] = data["delivery"]
        features["equipment"] = data["equipment"]

        features["month"] = date.dt.month.astype(int)
        features["day_of_week"] = date.dt.dayofweek.astype(int)
        features["day_of_year"] = date.dt.dayofyear.astype(int)
        features["week_of_year"] = date.dt.isocalendar().week.astype(int)
        features["days_since_2025"] = (date - pd.Timestamp("2025-01-01")).dt.days.astype(int)
        features["month_sin"] = np.sin(2.0 * np.pi * features["month"] / 12.0)
        features["month_cos"] = np.cos(2.0 * np.pi * features["month"] / 12.0)
        features["dow_sin"] = np.sin(2.0 * np.pi * features["day_of_week"] / 7.0)
        features["dow_cos"] = np.cos(2.0 * np.pi * features["day_of_week"] / 7.0)
        features["doy_sin"] = np.sin(2.0 * np.pi * features["day_of_year"] / 365.25)
        features["doy_cos"] = np.cos(2.0 * np.pi * features["day_of_year"] / 365.25)

        features["haversine_distance"] = self._haversine(features)
        features["distance_to_haversine_ratio"] = features["distance"] / np.maximum(
            features["haversine_distance"], 1.0
        )
        features["lat_delta"] = features["delivery_lat"] - features["pickup_lat"]
        features["lon_delta"] = features["delivery_lon"] - features["pickup_lon"]
        features["midpoint_lat"] = (features["pickup_lat"] + features["delivery_lat"]) / 2.0
        features["midpoint_lon"] = (features["pickup_lon"] + features["delivery_lon"]) / 2.0
        features["log_distance"] = np.log1p(features["distance"].clip(lower=0.0))
        features["log_weight"] = np.log1p(features["weight"].clip(lower=0.0))

        for column in CATEGORICAL_COLUMNS:
            features[column] = features[column].astype(str)

        if align and self.feature_columns_:
            features = features.reindex(columns=self.feature_columns_)
        return features

    def fit_transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        return self.fit(frame).transform(frame)
