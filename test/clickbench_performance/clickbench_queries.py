# =============================================================================
# Copyright 2026, Sirius Contributors.
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
# in compliance with the License. You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software distributed under the License
# is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
# or implied. See the License for the specific language governing permissions and limitations under
# the License.
# =============================================================================

"""ClickBench SQL (0-based) and source-column dependencies, in schema order.

QUERIES is loaded from the audited one-query-per-line queries.sql. QUERY_COLUMNS
lists base-table columns, including filters and sort keys. Q23 projects the full
schema, so the union is all hits columns, including OriginalURL.
"""

import re
from pathlib import Path

DIRECTORY = Path(__file__).resolve().parent
QUERIES = dict(enumerate((DIRECTORY / "queries.sql").read_text().splitlines()))
SCHEMA = (DIRECTORY / "create.sql").read_text()
SCHEMA_COLUMNS = [
    ("WatchID", "BIGINT"),
    ("JavaEnable", "SMALLINT"),
    ("Title", "TEXT"),
    ("GoodEvent", "SMALLINT"),
    ("EventTime", "TIMESTAMP"),
    ("EventDate", "DATE"),
    ("CounterID", "INTEGER"),
    ("ClientIP", "INTEGER"),
    ("RegionID", "INTEGER"),
    ("UserID", "BIGINT"),
    ("CounterClass", "SMALLINT"),
    ("OS", "SMALLINT"),
    ("UserAgent", "SMALLINT"),
    ("URL", "TEXT"),
    ("Referer", "TEXT"),
    ("IsRefresh", "SMALLINT"),
    ("RefererCategoryID", "SMALLINT"),
    ("RefererRegionID", "INTEGER"),
    ("URLCategoryID", "SMALLINT"),
    ("URLRegionID", "INTEGER"),
    ("ResolutionWidth", "SMALLINT"),
    ("ResolutionHeight", "SMALLINT"),
    ("ResolutionDepth", "SMALLINT"),
    ("FlashMajor", "SMALLINT"),
    ("FlashMinor", "SMALLINT"),
    ("FlashMinor2", "TEXT"),
    ("NetMajor", "SMALLINT"),
    ("NetMinor", "SMALLINT"),
    ("UserAgentMajor", "SMALLINT"),
    ("UserAgentMinor", "VARCHAR(255)"),
    ("CookieEnable", "SMALLINT"),
    ("JavascriptEnable", "SMALLINT"),
    ("IsMobile", "SMALLINT"),
    ("MobilePhone", "SMALLINT"),
    ("MobilePhoneModel", "TEXT"),
    ("Params", "TEXT"),
    ("IPNetworkID", "INTEGER"),
    ("TraficSourceID", "SMALLINT"),
    ("SearchEngineID", "SMALLINT"),
    ("SearchPhrase", "TEXT"),
    ("AdvEngineID", "SMALLINT"),
    ("IsArtifical", "SMALLINT"),
    ("WindowClientWidth", "SMALLINT"),
    ("WindowClientHeight", "SMALLINT"),
    ("ClientTimeZone", "SMALLINT"),
    ("ClientEventTime", "TIMESTAMP"),
    ("SilverlightVersion1", "SMALLINT"),
    ("SilverlightVersion2", "SMALLINT"),
    ("SilverlightVersion3", "INTEGER"),
    ("SilverlightVersion4", "SMALLINT"),
    ("PageCharset", "TEXT"),
    ("CodeVersion", "INTEGER"),
    ("IsLink", "SMALLINT"),
    ("IsDownload", "SMALLINT"),
    ("IsNotBounce", "SMALLINT"),
    ("FUniqID", "BIGINT"),
    ("OriginalURL", "TEXT"),
    ("HID", "INTEGER"),
    ("IsOldCounter", "SMALLINT"),
    ("IsEvent", "SMALLINT"),
    ("IsParameter", "SMALLINT"),
    ("DontCountHits", "SMALLINT"),
    ("WithHash", "SMALLINT"),
    ("HitColor", "CHAR"),
    ("LocalEventTime", "TIMESTAMP"),
    ("Age", "SMALLINT"),
    ("Sex", "SMALLINT"),
    ("Income", "SMALLINT"),
    ("Interests", "SMALLINT"),
    ("Robotness", "SMALLINT"),
    ("RemoteIP", "INTEGER"),
    ("WindowName", "INTEGER"),
    ("OpenerName", "INTEGER"),
    ("HistoryLength", "SMALLINT"),
    ("BrowserLanguage", "TEXT"),
    ("BrowserCountry", "TEXT"),
    ("SocialNetwork", "TEXT"),
    ("SocialAction", "TEXT"),
    ("HTTPError", "SMALLINT"),
    ("SendTiming", "INTEGER"),
    ("DNSTiming", "INTEGER"),
    ("ConnectTiming", "INTEGER"),
    ("ResponseStartTiming", "INTEGER"),
    ("ResponseEndTiming", "INTEGER"),
    ("FetchTiming", "INTEGER"),
    ("SocialSourceNetworkID", "SMALLINT"),
    ("SocialSourcePage", "TEXT"),
    ("ParamPrice", "BIGINT"),
    ("ParamOrderID", "TEXT"),
    ("ParamCurrency", "TEXT"),
    ("ParamCurrencyID", "SMALLINT"),
    ("OpenstatServiceName", "TEXT"),
    ("OpenstatCampaignID", "TEXT"),
    ("OpenstatAdID", "TEXT"),
    ("OpenstatSourceID", "TEXT"),
    ("UTMSource", "TEXT"),
    ("UTMMedium", "TEXT"),
    ("UTMCampaign", "TEXT"),
    ("UTMContent", "TEXT"),
    ("UTMTerm", "TEXT"),
    ("FromTag", "TEXT"),
    ("HasGCLID", "SMALLINT"),
    ("RefererHash", "BIGINT"),
    ("URLHash", "BIGINT"),
    ("CLID", "INTEGER"),
]
QUERY_COLUMNS = {
    0: frozenset([]),
    1: frozenset(["AdvEngineID"]),
    2: frozenset(["ResolutionWidth", "AdvEngineID"]),
    3: frozenset(["UserID"]),
    4: frozenset(["UserID"]),
    5: frozenset(["SearchPhrase"]),
    6: frozenset(["EventDate"]),
    7: frozenset(["AdvEngineID"]),
    8: frozenset(["RegionID", "UserID"]),
    9: frozenset(["RegionID", "UserID", "ResolutionWidth", "AdvEngineID"]),
    10: frozenset(["UserID", "MobilePhoneModel"]),
    11: frozenset(["UserID", "MobilePhone", "MobilePhoneModel"]),
    12: frozenset(["SearchPhrase"]),
    13: frozenset(["UserID", "SearchPhrase"]),
    14: frozenset(["SearchEngineID", "SearchPhrase"]),
    15: frozenset(["UserID"]),
    16: frozenset(["UserID", "SearchPhrase"]),
    17: frozenset(["UserID", "SearchPhrase"]),
    18: frozenset(["EventTime", "UserID", "SearchPhrase"]),
    19: frozenset(["UserID"]),
    20: frozenset(["URL"]),
    21: frozenset(["URL", "SearchPhrase"]),
    22: frozenset(["Title", "UserID", "URL", "SearchPhrase"]),
    23: frozenset(
        [
            "WatchID",
            "JavaEnable",
            "Title",
            "GoodEvent",
            "EventTime",
            "EventDate",
            "CounterID",
            "ClientIP",
            "RegionID",
            "UserID",
            "CounterClass",
            "OS",
            "UserAgent",
            "URL",
            "Referer",
            "IsRefresh",
            "RefererCategoryID",
            "RefererRegionID",
            "URLCategoryID",
            "URLRegionID",
            "ResolutionWidth",
            "ResolutionHeight",
            "ResolutionDepth",
            "FlashMajor",
            "FlashMinor",
            "FlashMinor2",
            "NetMajor",
            "NetMinor",
            "UserAgentMajor",
            "UserAgentMinor",
            "CookieEnable",
            "JavascriptEnable",
            "IsMobile",
            "MobilePhone",
            "MobilePhoneModel",
            "Params",
            "IPNetworkID",
            "TraficSourceID",
            "SearchEngineID",
            "SearchPhrase",
            "AdvEngineID",
            "IsArtifical",
            "WindowClientWidth",
            "WindowClientHeight",
            "ClientTimeZone",
            "ClientEventTime",
            "SilverlightVersion1",
            "SilverlightVersion2",
            "SilverlightVersion3",
            "SilverlightVersion4",
            "PageCharset",
            "CodeVersion",
            "IsLink",
            "IsDownload",
            "IsNotBounce",
            "FUniqID",
            "OriginalURL",
            "HID",
            "IsOldCounter",
            "IsEvent",
            "IsParameter",
            "DontCountHits",
            "WithHash",
            "HitColor",
            "LocalEventTime",
            "Age",
            "Sex",
            "Income",
            "Interests",
            "Robotness",
            "RemoteIP",
            "WindowName",
            "OpenerName",
            "HistoryLength",
            "BrowserLanguage",
            "BrowserCountry",
            "SocialNetwork",
            "SocialAction",
            "HTTPError",
            "SendTiming",
            "DNSTiming",
            "ConnectTiming",
            "ResponseStartTiming",
            "ResponseEndTiming",
            "FetchTiming",
            "SocialSourceNetworkID",
            "SocialSourcePage",
            "ParamPrice",
            "ParamOrderID",
            "ParamCurrency",
            "ParamCurrencyID",
            "OpenstatServiceName",
            "OpenstatCampaignID",
            "OpenstatAdID",
            "OpenstatSourceID",
            "UTMSource",
            "UTMMedium",
            "UTMCampaign",
            "UTMContent",
            "UTMTerm",
            "FromTag",
            "HasGCLID",
            "RefererHash",
            "URLHash",
            "CLID",
        ]
    ),
    24: frozenset(["EventTime", "SearchPhrase"]),
    25: frozenset(["SearchPhrase"]),
    26: frozenset(["EventTime", "SearchPhrase"]),
    27: frozenset(["CounterID", "URL"]),
    28: frozenset(["Referer"]),
    29: frozenset(["ResolutionWidth"]),
    30: frozenset(
        [
            "ClientIP",
            "IsRefresh",
            "ResolutionWidth",
            "SearchEngineID",
            "SearchPhrase",
        ]
    ),
    31: frozenset(
        ["WatchID", "ClientIP", "IsRefresh", "ResolutionWidth", "SearchPhrase"]
    ),
    32: frozenset(["WatchID", "ClientIP", "IsRefresh", "ResolutionWidth"]),
    33: frozenset(["URL"]),
    34: frozenset(["URL"]),
    35: frozenset(["ClientIP"]),
    36: frozenset(["EventDate", "CounterID", "URL", "IsRefresh", "DontCountHits"]),
    37: frozenset(["Title", "EventDate", "CounterID", "IsRefresh", "DontCountHits"]),
    38: frozenset(
        ["EventDate", "CounterID", "URL", "IsRefresh", "IsLink", "IsDownload"]
    ),
    39: frozenset(
        [
            "EventDate",
            "CounterID",
            "URL",
            "Referer",
            "IsRefresh",
            "TraficSourceID",
            "SearchEngineID",
            "AdvEngineID",
        ]
    ),
    40: frozenset(
        [
            "EventDate",
            "CounterID",
            "IsRefresh",
            "TraficSourceID",
            "RefererHash",
            "URLHash",
        ]
    ),
    41: frozenset(
        [
            "EventDate",
            "CounterID",
            "IsRefresh",
            "WindowClientWidth",
            "WindowClientHeight",
            "DontCountHits",
            "URLHash",
        ]
    ),
    42: frozenset(
        ["EventTime", "EventDate", "CounterID", "IsRefresh", "DontCountHits"]
    ),
}


def union_columns_by_table():
    union = set().union(*QUERY_COLUMNS.values())
    return {"hits": [name for name, _ in SCHEMA_COLUMNS if name in union]}


def sql_literal(value):
    return "'" + str(value).replace("'", "''") + "'"


def sql_identifier(value):
    return '"' + str(value).replace('"', '""') + '"'


def parquet_query(query, parquet):
    """Use the same Parquet file/glob identity for pinning and scans."""
    return render_query(query, parquet)


def render_query(query, parquet, source="parquet", table_catalog=None):
    """Render the shared SQL for GPU execution and the extension-free CPU oracle."""
    if source == "table":
        if table_catalog is None:
            return query
        replacement = f"{sql_identifier(table_catalog)}.main.hits"
    elif source == "parquet":
        replacement = f"read_parquet({sql_literal(parquet)})"
    else:
        raise ValueError(f"unknown source: {source}")
    return re.sub(
        r"\bFROM hits\b",
        lambda _: f"FROM {replacement}",
        query,
        flags=re.IGNORECASE,
    )
