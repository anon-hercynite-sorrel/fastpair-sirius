# ClickBench query provenance

Schema and original queries: `9f30b9ca^:scripts/clickbench_runner/`.
Effective queries: the 43 texts inside `gpu_processing` in
`9f30b9ca^:test/sql/clickbench-sirius.test`. Numbering is always 0-based.

All differences are listed below, including aliases that leave values unchanged.
Comparison preserves row order with numeric tolerance. Q23 orders by EventTime,
WatchID; if these keys are duplicated with differing projected values, ties remain
nondeterministic. Verbatim replicated rows are identical, so those ties are harmless.
Q7 still has no key after its count sort: equal counts can reorder distinct AdvEngineID
rows. The harness keeps the upstream text and reports validation failures; it does
not sort results or relax comparison globally.

## Query 0

Original:

```sql
SELECT COUNT(*) FROM hits;
```

Edited:

```sql
SELECT COUNT(*) AS total FROM hits;
```

Adopt upstream aliases; result values and order are unchanged.

## Query 1

Original:

```sql
SELECT COUNT(*) FROM hits WHERE AdvEngineID <> 0;
```

Edited:

```sql
SELECT COUNT(*) AS total FROM hits WHERE AdvEngineID <> 0;
```

Adopt upstream aliases; result values and order are unchanged.

## Query 2

Original:

```sql
SELECT SUM(AdvEngineID), COUNT(*), AVG(ResolutionWidth) FROM hits;
```

Edited:

```sql
SELECT SUM(AdvEngineID) AS sum, COUNT(*) AS count, AVG(ResolutionWidth) AS average FROM hits;
```

Adopt upstream aliases; result values and order are unchanged.

## Query 3

Original:

```sql
SELECT AVG(UserID) FROM hits;
```

Edited:

```sql
SELECT AVG(UserID) AS average FROM hits;
```

Adopt upstream aliases; result values and order are unchanged.

## Query 4

Original:

```sql
SELECT COUNT(DISTINCT UserID) FROM hits;
```

Edited:

```sql
SELECT COUNT(DISTINCT UserID) AS c_dist FROM hits;
```

Adopt upstream aliases; result values and order are unchanged.

## Query 5

Original:

```sql
SELECT COUNT(DISTINCT SearchPhrase) FROM hits;
```

Edited:

```sql
SELECT COUNT(DISTINCT SearchPhrase) AS c_dist FROM hits;
```

Adopt upstream aliases; result values and order are unchanged.

## Query 6

Original:

```sql
SELECT MIN(EventDate), MAX(EventDate) FROM hits;
```

Edited:

```sql
SELECT MIN(EventDate) AS min_date, MAX(EventDate) AS max_date FROM hits;
```

Adopt upstream aliases; result values and order are unchanged.

## Query 7

Original:

```sql
SELECT AdvEngineID, COUNT(*) FROM hits WHERE AdvEngineID <> 0 GROUP BY AdvEngineID ORDER BY COUNT(*) DESC;
```

Edited:

```sql
SELECT AdvEngineID, COUNT(*) AS count FROM hits WHERE AdvEngineID <> 0 GROUP BY AdvEngineID ORDER BY count DESC;
```

Adopt upstream ordering to break ties before LIMIT (or make grouped output deterministic), including upstream aliases.

## Query 8

Original:

```sql
SELECT RegionID, COUNT(DISTINCT UserID) AS u FROM hits GROUP BY RegionID ORDER BY u DESC LIMIT 10;
```

Edited:

```sql
SELECT RegionID, COUNT(DISTINCT UserID) AS u FROM hits GROUP BY RegionID ORDER BY u DESC, RegionID LIMIT 10;
```

Adopt upstream ordering to break ties before LIMIT (or make grouped output deterministic), including upstream aliases.

## Query 9

Original:

```sql
SELECT RegionID, SUM(AdvEngineID), COUNT(*) AS c, AVG(ResolutionWidth), COUNT(DISTINCT UserID) FROM hits GROUP BY RegionID ORDER BY c DESC LIMIT 10;
```

Edited:

```sql
SELECT RegionID, SUM(AdvEngineID) AS sum, COUNT(*) AS c, AVG(ResolutionWidth) AS avg, COUNT(DISTINCT UserID) AS c_dist FROM hits GROUP BY RegionID ORDER BY c DESC, RegionID LIMIT 10;
```

Adopt upstream ordering to break ties before LIMIT (or make grouped output deterministic), including upstream aliases.

## Query 10

Original:

```sql
SELECT MobilePhoneModel, COUNT(DISTINCT UserID) AS u FROM hits WHERE MobilePhoneModel <> '' GROUP BY MobilePhoneModel ORDER BY u DESC LIMIT 10;
```

Edited:

```sql
SELECT MobilePhoneModel, COUNT(DISTINCT UserID) AS u FROM hits WHERE MobilePhoneModel <> '' GROUP BY MobilePhoneModel ORDER BY u DESC, MobilePhoneModel DESC LIMIT 10;
```

Adopt upstream ordering to break ties before LIMIT (or make grouped output deterministic), including upstream aliases.

## Query 11

Original:

```sql
SELECT MobilePhone, MobilePhoneModel, COUNT(DISTINCT UserID) AS u FROM hits WHERE MobilePhoneModel <> '' GROUP BY MobilePhone, MobilePhoneModel ORDER BY u DESC LIMIT 10;
```

Edited:

```sql
SELECT MobilePhone, MobilePhoneModel, COUNT(DISTINCT UserID) AS u FROM hits WHERE MobilePhoneModel <> '' GROUP BY MobilePhone, MobilePhoneModel ORDER BY u DESC, MobilePhone, MobilePhoneModel LIMIT 10;
```

Adopt upstream ordering to break ties before LIMIT (or make grouped output deterministic), including upstream aliases.

## Query 12

Original:

```sql
SELECT SearchPhrase, COUNT(*) AS c FROM hits WHERE SearchPhrase <> '' GROUP BY SearchPhrase ORDER BY c DESC LIMIT 10;
```

Edited:

```sql
SELECT SearchPhrase, COUNT(*) AS c FROM hits WHERE SearchPhrase <> '' GROUP BY SearchPhrase ORDER BY c DESC, SearchPhrase LIMIT 10;
```

Adopt upstream ordering to break ties before LIMIT (or make grouped output deterministic), including upstream aliases.

## Query 13

Original:

```sql
SELECT SearchPhrase, COUNT(DISTINCT UserID) AS u FROM hits WHERE SearchPhrase <> '' GROUP BY SearchPhrase ORDER BY u DESC LIMIT 10;
```

Edited:

```sql
SELECT SearchPhrase, COUNT(DISTINCT UserID) AS u FROM hits WHERE SearchPhrase <> '' GROUP BY SearchPhrase ORDER BY u DESC, SearchPhrase LIMIT 10;
```

Adopt upstream ordering to break ties before LIMIT (or make grouped output deterministic), including upstream aliases.

## Query 14

Original:

```sql
SELECT SearchEngineID, SearchPhrase, COUNT(*) AS c FROM hits WHERE SearchPhrase <> '' GROUP BY SearchEngineID, SearchPhrase ORDER BY c DESC LIMIT 10;
```

Edited:

```sql
SELECT SearchEngineID, SearchPhrase, COUNT(*) AS c FROM hits WHERE SearchPhrase <> '' GROUP BY SearchEngineID, SearchPhrase ORDER BY c DESC, SearchEngineID, SearchPhrase LIMIT 10;
```

Adopt upstream ordering to break ties before LIMIT (or make grouped output deterministic), including upstream aliases.

## Query 15

Original:

```sql
SELECT UserID, COUNT(*) FROM hits GROUP BY UserID ORDER BY COUNT(*) DESC LIMIT 10;
```

Edited:

```sql
SELECT UserID, COUNT(*) AS count FROM hits GROUP BY UserID ORDER BY COUNT(*) DESC, UserID LIMIT 10;
```

Adopt upstream ordering to break ties before LIMIT (or make grouped output deterministic), including upstream aliases.

## Query 16

Original:

```sql
SELECT UserID, SearchPhrase, COUNT(*) FROM hits GROUP BY UserID, SearchPhrase ORDER BY COUNT(*) DESC LIMIT 10;
```

Edited:

```sql
SELECT UserID, SearchPhrase, COUNT(*) AS count FROM hits GROUP BY UserID, SearchPhrase ORDER BY COUNT(*) DESC, UserID, SearchPhrase LIMIT 10;
```

Adopt upstream ordering to break ties before LIMIT (or make grouped output deterministic), including upstream aliases.

## Query 17

Original:

```sql
SELECT UserID, SearchPhrase, COUNT(*) FROM hits GROUP BY UserID, SearchPhrase LIMIT 10;
```

Edited:

```sql
SELECT UserID, SearchPhrase, COUNT(*) AS count FROM hits GROUP BY UserID, SearchPhrase ORDER BY UserID, SearchPhrase, count LIMIT 10;
```

This variant adds an ordered top-k to Q17: the original grouped query used an
unordered LIMIT. Selecting the first ten groups in key order changes the work
performed, beyond making repeated results deterministic.

## Query 18

Original:

```sql
SELECT UserID, extract(minute FROM EventTime) AS m, SearchPhrase, COUNT(*) FROM hits GROUP BY UserID, m, SearchPhrase ORDER BY COUNT(*) DESC LIMIT 10;
```

Edited:

```sql
SELECT UserID, extract(minute FROM EventTime) AS m, SearchPhrase, COUNT(*) FROM hits GROUP BY UserID, m, SearchPhrase ORDER BY COUNT(*) DESC, m, SearchPhrase, UserID LIMIT 10;
```

Adopt upstream ordering to break ties before LIMIT (or make grouped output deterministic), including upstream aliases.

## Query 20

Original:

```sql
SELECT COUNT(*) FROM hits WHERE URL LIKE '%google%';
```

Edited:

```sql
SELECT COUNT(*) AS count FROM hits WHERE URL LIKE '%google%';
```

Adopt upstream aliases; result values and order are unchanged.

## Query 21

Original:

```sql
SELECT SearchPhrase, MIN(URL), COUNT(*) AS c FROM hits WHERE URL LIKE '%google%' AND SearchPhrase <> '' GROUP BY SearchPhrase ORDER BY c DESC LIMIT 10;
```

Edited:

```sql
SELECT SearchPhrase, MIN(URL) AS min, COUNT(*) AS c FROM hits WHERE URL LIKE '%google%' AND SearchPhrase <> '' GROUP BY SearchPhrase ORDER BY c DESC, SearchPhrase LIMIT 10;
```

Adopt upstream ordering to break ties before LIMIT (or make grouped output deterministic), including upstream aliases.

## Query 22

Original:

```sql
SELECT SearchPhrase, MIN(URL), MIN(Title), COUNT(*) AS c, COUNT(DISTINCT UserID) FROM hits WHERE Title LIKE '%Google%' AND URL NOT LIKE '%.google.%' AND SearchPhrase <> '' GROUP BY SearchPhrase ORDER BY c DESC LIMIT 10;
```

Edited:

```sql
SELECT SearchPhrase, MIN(URL) AS min_url, MIN(Title) AS min_title, COUNT(*) AS c, COUNT(DISTINCT UserID) AS c_distinct FROM hits WHERE Title LIKE '%Google%' AND URL NOT LIKE '%.google.%' AND SearchPhrase <> '' GROUP BY SearchPhrase ORDER BY c DESC, SearchPhrase, min_url, min_title LIMIT 10;
```

Adopt upstream ordering to break ties before LIMIT (or make grouped output deterministic), including upstream aliases.

## Query 23

Original:

```sql
SELECT * FROM hits WHERE URL LIKE '%google%' ORDER BY EventTime LIMIT 10;
```

Edited:

```sql
SELECT * FROM hits WHERE URL LIKE '%google%' ORDER BY EventTime, WatchID LIMIT 10;
```

Adopt upstream ordering to break ties before LIMIT (or make grouped output deterministic), including upstream aliases.

## Query 24

Original:

```sql
SELECT SearchPhrase FROM hits WHERE SearchPhrase <> '' ORDER BY EventTime LIMIT 10;
```

Edited:

```sql
SELECT SearchPhrase FROM hits WHERE SearchPhrase <> '' ORDER BY EventTime, SearchPhrase LIMIT 10;
```

Adopt upstream ordering to break ties before LIMIT (or make grouped output deterministic), including upstream aliases.

Q24 is identical to Q26 in this variant. Including both in the 43-query geomean
weights this workload twice; this is a property of the chosen variant.

## Query 27

Original:

```sql
SELECT CounterID, AVG(STRLEN(URL)) AS l, COUNT(*) AS c FROM hits WHERE URL <> '' GROUP BY CounterID HAVING COUNT(*) > 100000 ORDER BY l DESC LIMIT 25;
```

Edited:

```sql
SELECT CounterID, AVG(STRLEN(URL)) AS l, COUNT(*) AS c FROM hits WHERE URL <> '' GROUP BY CounterID HAVING COUNT(*) > 100000 ORDER BY l DESC, CounterID LIMIT 25;
```

Adopt upstream ordering to break ties before LIMIT (or make grouped output deterministic), including upstream aliases.

## Query 28

Original:

```sql
SELECT REGEXP_REPLACE(Referer, '^https?://(?:www\.)?([^/]+)/.*$', '\1') AS k, AVG(STRLEN(Referer)) AS l, COUNT(*) AS c, MIN(Referer) FROM hits WHERE Referer <> '' GROUP BY k HAVING COUNT(*) > 100000 ORDER BY l DESC LIMIT 25;
```

Edited:

```sql
SELECT REGEXP_REPLACE(Referer, '^https?://(?:www\.)?([^/]+)/.*$', '\1') AS k, AVG(STRLEN(Referer)) AS l, COUNT(*) AS c, MIN(Referer) AS min FROM hits WHERE Referer <> '' GROUP BY k HAVING COUNT(*) > 100000 ORDER BY l DESC, k LIMIT 25;
```

Adopt upstream ordering to break ties before LIMIT (or make grouped output deterministic), including upstream aliases.

## Query 30

Original:

```sql
SELECT SearchEngineID, ClientIP, COUNT(*) AS c, SUM(IsRefresh), AVG(ResolutionWidth) FROM hits WHERE SearchPhrase <> '' GROUP BY SearchEngineID, ClientIP ORDER BY c DESC LIMIT 10;
```

Edited:

```sql
SELECT SearchEngineID, ClientIP, COUNT(*) AS c, SUM(IsRefresh) AS sum, AVG(ResolutionWidth) AS avg FROM hits WHERE SearchPhrase <> '' GROUP BY SearchEngineID, ClientIP ORDER BY c DESC, SearchEngineID, ClientIP, avg, sum LIMIT 10;
```

Adopt upstream ordering to break ties before LIMIT (or make grouped output deterministic), including upstream aliases.

## Query 31

Original:

```sql
SELECT WatchID, ClientIP, COUNT(*) AS c, SUM(IsRefresh), AVG(ResolutionWidth) FROM hits WHERE SearchPhrase <> '' GROUP BY WatchID, ClientIP ORDER BY c DESC LIMIT 10;
```

Edited:

```sql
SELECT WatchID, ClientIP, COUNT(*) AS c, SUM(IsRefresh) AS sum, AVG(ResolutionWidth) AS avg FROM hits WHERE SearchPhrase <> '' GROUP BY WatchID, ClientIP ORDER BY c DESC, ClientIP, WatchID, sum, avg LIMIT 10;
```

Adopt upstream ordering to break ties before LIMIT (or make grouped output deterministic), including upstream aliases.

## Query 32

Original:

```sql
SELECT WatchID, ClientIP, COUNT(*) AS c, SUM(IsRefresh), AVG(ResolutionWidth) FROM hits GROUP BY WatchID, ClientIP ORDER BY c DESC LIMIT 10;
```

Edited:

```sql
SELECT WatchID, ClientIP, COUNT(*) AS c, SUM(IsRefresh) AS sum, AVG(ResolutionWidth) AS avg FROM hits GROUP BY WatchID, ClientIP ORDER BY c DESC, WatchID, ClientIP, avg, sum LIMIT 10;
```

Adopt upstream ordering to break ties before LIMIT (or make grouped output deterministic), including upstream aliases.

## Query 33

Original:

```sql
SELECT URL, COUNT(*) AS c FROM hits GROUP BY URL ORDER BY c DESC LIMIT 10;
```

Edited:

```sql
SELECT URL, COUNT(*) AS c FROM hits GROUP BY URL ORDER BY c DESC, URL LIMIT 10;
```

Adopt upstream ordering to break ties before LIMIT (or make grouped output deterministic), including upstream aliases.

## Query 34

Original:

```sql
SELECT 1, URL, COUNT(*) AS c FROM hits GROUP BY 1, URL ORDER BY c DESC LIMIT 10;
```

Edited:

```sql
SELECT 1, URL, COUNT(*) AS c FROM hits GROUP BY 1, URL ORDER BY c DESC, URL LIMIT 10;
```

Adopt upstream ordering to break ties before LIMIT (or make grouped output deterministic), including upstream aliases.

## Query 35

Original:

```sql
SELECT ClientIP, ClientIP - 1, ClientIP - 2, ClientIP - 3, COUNT(*) AS c FROM hits GROUP BY ClientIP, ClientIP - 1, ClientIP - 2, ClientIP - 3 ORDER BY c DESC LIMIT 10;
```

Edited:

```sql
SELECT ClientIP, ClientIP - 1, ClientIP - 2, ClientIP - 3, COUNT(*) AS c FROM hits GROUP BY ClientIP, ClientIP - 1, ClientIP - 2, ClientIP - 3 ORDER BY c DESC, ClientIP LIMIT 10;
```

Adopt upstream ordering to break ties before LIMIT (or make grouped output deterministic), including upstream aliases.

## Query 36

Original:

```sql
SELECT URL, COUNT(*) AS PageViews FROM hits WHERE CounterID = 62 AND EventDate >= '2013-07-01' AND EventDate <= '2013-07-31' AND DontCountHits = 0 AND IsRefresh = 0 AND URL <> '' GROUP BY URL ORDER BY PageViews DESC LIMIT 10;
```

Edited:

```sql
SELECT URL, COUNT(*) AS PageViews FROM hits WHERE CounterID = 62 AND EventDate >= '2013-07-01' AND EventDate <= '2013-07-31' AND DontCountHits = 0 AND IsRefresh = 0 AND URL <> '' GROUP BY URL ORDER BY PageViews DESC, URL LIMIT 10;
```

Adopt upstream ordering to break ties before LIMIT (or make grouped output deterministic), including upstream aliases.

## Query 37

Original:

```sql
SELECT Title, COUNT(*) AS PageViews FROM hits WHERE CounterID = 62 AND EventDate >= '2013-07-01' AND EventDate <= '2013-07-31' AND DontCountHits = 0 AND IsRefresh = 0 AND Title <> '' GROUP BY Title ORDER BY PageViews DESC LIMIT 10;
```

Edited:

```sql
SELECT Title, COUNT(*) AS PageViews FROM hits WHERE CounterID = 62 AND EventDate >= '2013-07-01' AND EventDate <= '2013-07-31' AND DontCountHits = 0 AND IsRefresh = 0 AND Title <> '' GROUP BY Title ORDER BY PageViews DESC, Title LIMIT 10;
```

Adopt upstream ordering to break ties before LIMIT (or make grouped output deterministic), including upstream aliases.

## Query 38

Original:

```sql
SELECT URL, COUNT(*) AS PageViews FROM hits WHERE CounterID = 62 AND EventDate >= '2013-07-01' AND EventDate <= '2013-07-31' AND IsRefresh = 0 AND IsLink <> 0 AND IsDownload = 0 GROUP BY URL ORDER BY PageViews DESC LIMIT 10 OFFSET 1000;
```

Edited:

```sql
SELECT URL, COUNT(*) AS PageViews FROM hits WHERE CounterID = 62 AND EventDate >= '2013-07-01' AND EventDate <= '2013-07-31' AND IsRefresh = 0 AND IsLink <> 0 AND IsDownload = 0 GROUP BY URL ORDER BY PageViews DESC, URL LIMIT 10 OFFSET 1000;
```

Adopt upstream ordering to break ties before LIMIT (or make grouped output deterministic), including upstream aliases.

## Query 39

Original:

```sql
SELECT TraficSourceID, SearchEngineID, AdvEngineID, CASE WHEN (SearchEngineID = 0 AND AdvEngineID = 0) THEN Referer ELSE '' END AS Src, URL AS Dst, COUNT(*) AS PageViews FROM hits WHERE CounterID = 62 AND EventDate >= '2013-07-01' AND EventDate <= '2013-07-31' AND IsRefresh = 0 GROUP BY TraficSourceID, SearchEngineID, AdvEngineID, Src, Dst ORDER BY PageViews DESC LIMIT 10 OFFSET 1000;
```

Edited:

```sql
SELECT TraficSourceID, SearchEngineID, AdvEngineID, CASE WHEN (SearchEngineID = 0 AND AdvEngineID = 0) THEN Referer ELSE '' END AS Src, URL AS Dst, COUNT(*) AS PageViews FROM hits WHERE CounterID = 62 AND EventDate >= '2013-07-01' AND EventDate <= '2013-07-31' AND IsRefresh = 0 GROUP BY TraficSourceID, SearchEngineID, AdvEngineID, Src, Dst ORDER BY PageViews DESC, TraficSourceID, SearchEngineID, AdvEngineID, Src, Dst LIMIT 10 OFFSET 1000;
```

Adopt upstream ordering to break ties before LIMIT (or make grouped output deterministic), including upstream aliases.

## Query 40

Original:

```sql
SELECT URLHash, EventDate, COUNT(*) AS PageViews FROM hits WHERE CounterID = 62 AND EventDate >= '2013-07-01' AND EventDate <= '2013-07-31' AND IsRefresh = 0 AND TraficSourceID IN (-1, 6) AND RefererHash = 3594120000172545465 GROUP BY URLHash, EventDate ORDER BY PageViews DESC LIMIT 10 OFFSET 100;
```

Edited:

```sql
SELECT URLHash, EventDate, COUNT(*) AS PageViews FROM hits WHERE CounterID = 62 AND EventDate >= '2013-07-01' AND EventDate <= '2013-07-31' AND IsRefresh = 0 AND TraficSourceID IN (-1, 6) AND RefererHash = 3594120000172545465 GROUP BY URLHash, EventDate ORDER BY PageViews DESC, URLHash, EventDate LIMIT 10 OFFSET 100;
```

Adopt upstream ordering to break ties before LIMIT (or make grouped output deterministic), including upstream aliases.

## Query 41

Original:

```sql
SELECT WindowClientWidth, WindowClientHeight, COUNT(*) AS PageViews FROM hits WHERE CounterID = 62 AND EventDate >= '2013-07-01' AND EventDate <= '2013-07-31' AND IsRefresh = 0 AND DontCountHits = 0 AND URLHash = 2868770270353813622 GROUP BY WindowClientWidth, WindowClientHeight ORDER BY PageViews DESC LIMIT 10 OFFSET 10000;
```

Edited:

```sql
SELECT WindowClientWidth, WindowClientHeight, COUNT(*) AS PageViews FROM hits WHERE CounterID = 62 AND EventDate >= '2013-07-01' AND EventDate <= '2013-07-31' AND IsRefresh = 0 AND DontCountHits = 0 AND URLHash = 2868770270353813622 GROUP BY WindowClientWidth, WindowClientHeight ORDER BY PageViews DESC, WindowClientHeight, WindowClientWidth LIMIT 10 OFFSET 10000;
```

Adopt upstream ordering to break ties before LIMIT (or make grouped output deterministic), including upstream aliases.
