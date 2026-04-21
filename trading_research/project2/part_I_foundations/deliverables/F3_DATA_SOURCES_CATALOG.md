# Data Sources Catalog
**Part I Foundations Task:** F-3
**Machine:** Gamma (cataloged from Omega)
**Status:** Complete
**Produced:** 2025-07-21
**Dependencies:** None (Wave 1)
**Feeds into:** F-4 (Asset + Timeframe Selection), F-5 (Data Pipeline Specification)

---

## Purpose

Comprehensive catalog of candidate data sources for Project 2. This is analysis only — no data has been acquired. Each source is evaluated on cost, quality, historical depth, API access, and Project 2 relevance. The catalog informs asset/timeframe selection (F-4) and pipeline design (F-5).

---

## Category 1: FX Price Data

### 1.1 OANDA (Project 1 Source)
- **URL:** https://developer.oanda.com/rest-live-v20/
- **Data types:** FX spot (70+ pairs), CFDs on indices/commodities
- **Cost tier:** Free with funded account (demo account also available)
- **Historical coverage:** ~20 years of daily OHLCV; granularity from 5-second to monthly candles
- **API availability:** Yes — REST v20 API. Key endpoints: `GET /v3/accounts/{id}/instruments/{instrument}/candles` for historical data; `GET /v3/accounts/{id}/pricing/stream` for real-time streaming (max 4 prices/sec per instrument)
- **Update frequency:** Real-time streaming; candles available at multiple granularities (S5, S10, S15, S30, M1, M2, M4, M5, M10, M15, M30, H1, H2, H3, H4, H6, H8, H12, D, W, M)
- **Quality assessment:** Institutional-grade. Already validated in Project 1 (23 years EUR/USD). Regulated by CFTC (US), FCA (UK), ASIC (AU), MAS (SG). Mid-price and bid/ask available. Known issue: candle count limit per request (~5000), requires pagination for long histories.
- **Project 2 relevance:** **HIGH** — Primary source, proven in Project 1. Continuity advantage. All existing pipelines (feature-eng, preprocessor, predictor) already handle OANDA CSV format.
- **Notes:** Project 1 used OANDA-sourced EUR/USD M1 data aggregated to 4H. API rate limits exist but are generous for research. Demo accounts give full API access with simulated pricing.

### 1.2 Dukascopy (Swiss Bank)
- **URL:** https://www.dukascopy.com/swiss/english/marketwatch/historical/
- **Data types:** FX (50+ pairs), commodities, indices, CFDs, crypto
- **Cost tier:** Free historical data download (demo account registration required); live trading account needed for real-time API
- **Historical coverage:** Tick data back to ~2003 for major FX pairs; M1 bars back further. Exact depth varies by instrument.
- **API availability:** JForex API (Java-based), FIX API for institutional clients. Historical Data Feed is browser-based download tool with export functionality. No simple REST API for historical data — programmatic access requires JForex SDK or third-party scrapers (e.g., `duka` Python package).
- **Update frequency:** Tick-level for live; historical exports updated periodically
- **Quality assessment:** Swiss FINMA-regulated bank. High reputation for data quality. ECN pricing (multiple liquidity providers). Tick data is genuine market data, not interpolated. Won "Best Fintech Forex Broker 2026", "Best Online Bank Switzerland 2024".
- **Project 2 relevance:** **HIGH** — Best free source for tick-level FX data. Essential if Project 2 experiments at sub-M1 resolution. Also useful for cross-validation against OANDA data.
- **Notes:** Programmatic download is non-trivial (browser-based tool, or JForex API). Community tools exist (`duka`, `dukascopy-node`) but may break with site changes. Worth investigating for tick-level experiments in Parts III-IV.

### 1.3 HistData.com
- **URL:** https://www.histdata.com/download-free-forex-data/
- **Data types:** FX only (major and minor pairs)
- **Cost tier:** Free download; paid auto-updates via Google Drive ($7/month for MT4/5 format)
- **Historical coverage:** M1 bars and tick data (1-second resolution) for major pairs going back to ~2000. Last data update confirmed: 2026-04-12.
- **API availability:** No REST API. Manual download or FTP/SFTP access (paid). Formats: MetaTrader 4/5, Generic ASCII (CSV), NinjaTrader, MetaStock, Excel.
- **Update frequency:** Periodic manual updates (last confirmed active April 2026). Google Drive subscribers get auto-updates.
- **Quality assessment:** Well-known in retail algo trading community. Data is aggregated from multiple sources. Quality is acceptable for M1-level research but not institutional-grade. Some gaps reported in older data. Blog posts stopped November 2022, but data files are still being updated.
- **Project 2 relevance:** **MEDIUM** — Useful as secondary free FX source for cross-validation. Generic ASCII format easily parseable. Not a primary source but good for sanity checks.
- **Notes:** Generic ASCII tick data format provides bid/ask columns. Free download requires navigating per-pair/per-year pages. Bulk download only via paid FTP.

### 1.4 TrueFX
- **URL:** https://www.truefx.com/
- **Data types:** FX spot (institutional tick-by-tick from tier-one banks)
- **Cost tier:** Professional: $4,950/month; Institutional: $7,450/month. Previously had a free historical data download option — unclear if still available.
- **Historical coverage:** Deep for major pairs (exact depth unknown without account)
- **API availability:** FIX protocol (event-driven streaming). Low-latency cross-connects in NY, LDN, TK for institutional plan.
- **Update frequency:** Real-time tick streaming
- **Quality assessment:** Institutional-grade. Prices directly from tier-one banks with no intermediary. 3-level depth of book on Institutional plan. Trademark of Integral Development Corp.
- **Project 2 relevance:** **LOW** — Far exceeds budget for a research project. Catalog for completeness. The free historical data download that previously existed may still be accessible but wasn't confirmed in current research.
- **Notes:** If the free historical tick download is still available, relevance jumps to HIGH. Worth checking periodically. The quality is among the best available for FX tick data.

### 1.5 Alpha Vantage (FX)
- **URL:** https://www.alphavantage.co/documentation/#fx
- **Data types:** FX pairs (physical currencies), crypto exchange rates
- **Cost tier:** Free tier (25 req/day); Premium plans for intraday FX data (`FX_INTRADAY` is premium). `FX_DAILY`, `FX_WEEKLY`, `FX_MONTHLY` are free.
- **Historical coverage:** FX_DAILY covers 20+ years with `outputsize=full`. Intraday limited to recent data on free tier.
- **API availability:** Yes — simple REST API. Endpoints: `FX_DAILY`, `FX_WEEKLY`, `FX_MONTHLY` (free); `FX_INTRADAY` at 1/5/15/30/60min (premium); `CURRENCY_EXCHANGE_RATE` for spot (free).
- **Update frequency:** Real-time for spot rates; daily for daily OHLC
- **Quality assessment:** Widely used retail API. Data sourced from forex markets. Not as precise as OANDA or Dukascopy for tick-level work. Good for daily+ timeframes.
- **Project 2 relevance:** **MEDIUM** — Free daily FX data is useful backup. Intraday requires premium. Main value: easy API for quick prototyping and cross-validation.
- **Notes:** Free API key is rate-limited (25 calls/day standard, 500 for premium basic). Also provides technical indicators pre-calculated (SMA, EMA, RSI, MACD, BBANDS, STOCH, etc.) — could be useful for feature engineering comparison.

### 1.6 Interactive Brokers (IB)
- **URL:** https://interactivebrokers.com/
- **Data types:** FX, equities, futures, options, bonds, crypto — one of the broadest instrument universes
- **Cost tier:** Requires funded trading account (min varies by account type). Data is essentially free with account. Market data subscriptions may apply.
- **Historical coverage:** Varies by instrument. FX: several years of M1 data. Limited to ~1 year of tick data via API. Historical data requests have pacing limits.
- **API availability:** TWS API (Python, Java, C++, C#), REST API via Client Portal. WebSocket streaming.
- **Update frequency:** Real-time with market data subscription
- **Quality assessment:** Institutional broker. Data quality is high but the API is notoriously complex and finicky. Rate limits on historical data requests. TWS must be running.
- **Project 2 relevance:** **LOW** — Only relevant if user has an IB account. API complexity is a significant friction point. Better alternatives exist for FX research data.
- **Notes:** Cataloged for completeness. If user already has an IB account, could be useful for multi-asset experiments (equities + FX).

---

## Category 2: Economic Calendar & Macro Data

### 2.1 FRED (Federal Reserve Economic Data)
- **URL:** https://fred.stlouisfed.org/ (direct); also accessible via Alpha Vantage API
- **Data types:** 800,000+ time series. US macro: GDP, CPI, unemployment, NFP, fed funds rate, treasury yields, retail sales, durable goods, housing starts, industrial production, etc. International data via IMF/World Bank pass-through.
- **Cost tier:** Free (public domain US government data)
- **Historical coverage:** Decades for most series (some back to 1940s+). Monthly/quarterly/annual depending on indicator.
- **API availability:** Yes — FRED API (free key required). Also available indirectly through Alpha Vantage's Economic Indicators endpoints: `REAL_GDP`, `CPI`, `INFLATION`, `UNEMPLOYMENT`, `NONFARM_PAYROLL`, `FEDERAL_FUNDS_RATE`, `TREASURY_YIELD`, `RETAIL_SALES`, `DURABLES`.
- **Update frequency:** Series-dependent. NFP: monthly (first Friday). CPI: monthly. GDP: quarterly with revisions.
- **Quality assessment:** Gold standard for US economic data. Primary source used by researchers, central banks, and financial institutions. Data is revised retroactively (important for backtesting: must use vintage data to avoid look-ahead bias).
- **Project 2 relevance:** **HIGH** — Essential for any event-driven or macro-informed strategy. NFP, CPI, and Fed decisions are among the highest-impact FX movers. Already accessible through Alpha Vantage's free tier.
- **Notes:** **Critical consideration for Project 2:** FRED provides point-in-time releases but not the *exact timestamp* of intraday releases. For event-driven strategies, need to supplement with an economic calendar that provides release times (see ForexFactory, TradingEconomics). Also: vintage data matters — using final revised values in backtest instead of first-release values is a form of look-ahead bias.

### 2.2 Trading Economics
- **URL:** https://tradingeconomics.com/api
- **Data types:** Economic indicators (20 million+ series for 196 countries), economic calendar, market data (FX, stocks, commodities, bonds), forecasts, financial statements
- **Cost tier:** Freemium API. Free: limited access. Paid plans: pricing not public (contact for quotes). Web scraping prohibited.
- **Historical coverage:** Deep historical data for most indicators. Calendar events with historical actuals/forecasts/previous.
- **API availability:** Yes — REST API. Modules: Indicators, Calendar, Markets, Financials, Forecasts. SDKs: Python, R, C#, Node.js, Java, PHP.
- **Update frequency:** Real-time calendar updates. Live market quotes.
- **Quality assessment:** One of the best aggregated macro data sources. Calendar is widely regarded as the best free economic calendar on the web. Data sourced from official statistical agencies.
- **Project 2 relevance:** **HIGH** — Calendar API is critical for event-driven strategies. Provides actual/forecast/previous triplets needed for surprise calculation. 196-country coverage enables cross-country macro analysis.
- **Notes:** API pricing needs investigation. Free tier may be too limited for systematic research. Calendar data with timestamps is the primary value for Project 2.

### 2.3 ForexFactory
- **URL:** https://www.forexfactory.com/calendar
- **Data types:** Economic calendar (FX-focused). Events with impact rating (high/medium/low), actual/forecast/previous values.
- **Cost tier:** Free (web access). No official API.
- **Historical coverage:** Calendar events going back many years. Historical data accessible via HTML.
- **API availability:** No official API. Community-built scrapers exist. Data extraction requires web scraping (check ToS).
- **Update frequency:** Real-time during event releases
- **Quality assessment:** De facto standard calendar in retail FX community. Impact ratings are useful for filtering. Data format is well-understood.
- **Project 2 relevance:** **MEDIUM** — Valuable calendar data but no official API means scraping risk. Better to use Trading Economics or FRED API for programmatic access. ForexFactory as manual cross-reference.
- **Notes:** Several open-source Python scrapers exist on GitHub. Community has reverse-engineered data URLs. Legal status of scraping unclear — use with caution.

### 2.4 Investing.com
- **URL:** https://www.investing.com/economic-calendar/
- **Data types:** Economic calendar, real-time market data (FX, crypto, stocks, indices, commodities, bonds), technical analysis, news
- **Cost tier:** Free website access; API access requires paid subscription (Investing.com API or via Fusion Media)
- **Historical coverage:** Extensive calendar and market data history
- **API availability:** Paid API only. No free programmatic access. Web scraping explicitly prohibited in ToS.
- **Update frequency:** Real-time
- **Quality assessment:** One of the most comprehensive financial data websites. Calendar is feature-rich. Data quality is good.
- **Project 2 relevance:** **LOW** — No free API kills programmatic access. Manual reference only. Prefer Trading Economics or FRED.
- **Notes:** Useful for manual research and cross-validation only.

### 2.5 ECB Statistical Data Warehouse
- **URL:** https://sdw.ecb.europa.eu/
- **Data types:** Euro area monetary, financial, and economic statistics. FX reference rates, interest rates, balance of payments, money supply, banking statistics.
- **Cost tier:** Free (public institution data)
- **Historical coverage:** Decades of Euro area data. ECB reference rates from 1999.
- **API availability:** Yes — SDMX REST API (Statistical Data and Metadata eXchange). Returns XML/JSON/CSV.
- **Update frequency:** Daily for reference rates, monthly/quarterly for most macro series
- **Quality assessment:** Official source for Euro area data. Authoritative and well-maintained. Reference rates are the official ECB daily fixing.
- **Project 2 relevance:** **MEDIUM** — Useful for EUR-denominated research. ECB reference rates can cross-validate broker data. Euro area macro data essential if expanding beyond US-centric indicators.
- **Notes:** SDMX API is standard but has a learning curve. Data warehouse web interface is clunky.

### 2.6 Alpha Vantage (Economic Indicators)
- **URL:** https://www.alphavantage.co/documentation/#economic-indicators
- **Data types:** US economic indicators sourced from FRED: Real GDP, GDP per capita, Treasury yields (3M-30Y), Federal Funds Rate, CPI, Inflation, Retail Sales, Durables, Unemployment, Nonfarm Payroll
- **Cost tier:** Free (included with any API key)
- **Historical coverage:** Decades (same as FRED underlying data)
- **API availability:** Yes — simple REST API. Endpoints: `REAL_GDP`, `REAL_GDP_PER_CAPITA`, `TREASURY_YIELD`, `FEDERAL_FUNDS_RATE`, `CPI`, `INFLATION`, `RETAIL_SALES`, `DURABLES`, `UNEMPLOYMENT`, `NONFARM_PAYROLL`. Also: `GOLD_SILVER_HISTORY`, `WTI`, `BRENT`, `NATURAL_GAS`, `COPPER`, `ALUMINUM`, `WHEAT`, `CORN`, `COTTON`, `SUGAR`, `COFFEE`, `ALL_COMMODITIES`.
- **Update frequency:** Series-dependent (same as FRED)
- **Quality assessment:** Convenient wrapper around FRED data. Data is identical to FRED. Bound by FRED Terms of Use.
- **Project 2 relevance:** **HIGH** — Easiest way to programmatically access major US macro indicators. Single API for FX + macro + commodities is convenient for pipeline simplicity.
- **Notes:** Not a separate source — this is FRED data via Alpha Vantage. Listed separately because the access method is different and more convenient than FRED's own API.

---

## Category 3: Alternative Data (per Jansen Framework)

### 3.1 NASDAQ Data Link (formerly Quandl)
- **URL:** https://data.nasdaq.com/
- **Data types:** 250+ datasets spanning market data, alternative data, ESG, economic indicators. Includes: Sharadar fundamentals, Zacks, commodity futures, sentiment indices, and more.
- **Cost tier:** Freemium. Some datasets free, many require paid subscription. Premium datasets: price varies by dataset.
- **Historical coverage:** Dataset-dependent. Some go back decades.
- **API availability:** Yes — REST API. Python (`nasdaqdatalink` package), R, Ruby, Excel plugins. Bulk download available.
- **Update frequency:** Dataset-dependent (daily to quarterly)
- **Quality assessment:** Acquired by NASDAQ. Professional-grade data marketplace. Previously Quandl, which was the go-to for quantitative research. Datasets are curated and documented.
- **Project 2 relevance:** **MEDIUM** — Valuable if specific alternative datasets are needed (e.g., CoT reports, sentiment indices). Not a primary FX data source. Worth exploring for macro factors.
- **Notes:** Free tier provides access to several useful datasets (e.g., CFTC Commitments of Traders). Premium datasets may be relevant for Part IV (ML features). Account required for API access.

### 3.2 Google Trends
- **URL:** https://trends.google.com/
- **Data types:** Search interest indices for any keyword/topic, by region and time. Relative scale (0-100).
- **Cost tier:** Free
- **Historical coverage:** From 2004 to present. Resolution: weekly for >5 years, daily for <90 days, hourly for <7 days.
- **API availability:** No official API. Unofficial: `pytrends` Python library (well-maintained, widely used). Rate-limited.
- **Update frequency:** Near-real-time (hourly granularity available for recent data)
- **Quality assessment:** Novel data source with documented alpha in academic literature (e.g., Preis et al. 2013 — Google Trends predicts stock market moves). Index is relative, not absolute, which complicates time-series analysis. Data revisions can occur.
- **Project 2 relevance:** **SPECULATIVE** — Interesting for experimental features (e.g., "EUR USD forecast" search trends as sentiment proxy). Not a core data source. Worth a small experiment in Part IV if time permits.
- **Notes:** Documented in Jansen's ML for Trading book as an alternative data source. `pytrends` is the de facto access method. Subject to Google's rate limiting and occasional API changes.

### 3.3 News Sentiment (Alpha Vantage)
- **URL:** https://www.alphavantage.co/documentation/#news-sentiment
- **Data types:** Market news articles with AI-generated sentiment scores. Topics: economy, technology, finance, real estate, etc. Per-ticker relevance and sentiment scoring.
- **Cost tier:** Free with API key
- **Historical coverage:** Recent articles (exact depth not documented, likely months not years)
- **API availability:** Yes — `NEWS_SENTIMENT` endpoint. Filterable by tickers, topics, time range. Returns: title, URL, source, summary, sentiment score (-1 to 1), relevance score.
- **Update frequency:** Near-real-time article ingestion
- **Quality assessment:** AI-generated sentiment — quality of NLP model unknown. Useful as a feature input, not as ground truth. Coverage biased toward US equities and English-language sources.
- **Project 2 relevance:** **LOW** — FX sentiment is poorly covered by equity-focused news APIs. More useful for equity strategies. Catalog for completeness.
- **Notes:** Could supplement with specialized FX news sources. Jansen discusses NLP-based sentiment as an alternative data category.

### 3.4 CFTC Commitments of Traders (CoT)
- **URL:** https://www.cftc.gov/MarketReports/CommitmentsofTraders/index.htm
- **Data types:** Weekly positioning data for futures markets. Net long/short positions by trader category (commercial, non-commercial, non-reportable). Includes FX futures (EUR, GBP, JPY, CHF, CAD, AUD, NZD, MXN).
- **Cost tier:** Free (US government data)
- **Historical coverage:** From 1986 to present (weekly)
- **API availability:** Bulk CSV download. Also available via NASDAQ Data Link. No real-time API (published weekly, Fridays for Tuesday data).
- **Update frequency:** Weekly (3-day lag)
- **Quality assessment:** Authoritative — official CFTC data. Widely used in FX macro analysis. The "speculative positioning" signal (non-commercial net longs) is a documented FX factor.
- **Project 2 relevance:** **HIGH** — One of the most studied alternative data sources for FX. Net speculative positioning is a known predictor of medium-term FX trends. Weekly frequency aligns well with adaptive re-optimization windows.
- **Notes:** Key limitation: 3-day publication lag. Data reflects positions as of Tuesday, published Friday after market close. For weekly+ strategies, this lag is manageable. For intraday, irrelevant.

### 3.5 Social Media / X (Twitter) API
- **URL:** https://developer.x.com/
- **Data types:** Social media posts, trending topics, volume metrics
- **Cost tier:** Basic: free (limited). Pro: $100/month. Enterprise: $42,000+/month.
- **Historical coverage:** Recent only on free/Pro tiers. Historical full archive on Enterprise.
- **API availability:** REST + Streaming APIs. v2 endpoints.
- **Update frequency:** Real-time streaming available
- **Quality assessment:** High noise-to-signal ratio. Requires significant NLP preprocessing. Academic literature shows some predictive value for FX/equity sentiment.
- **Project 2 relevance:** **SPECULATIVE** — High cost for useful tiers, high noise, uncertain signal quality for FX. Not a priority for Part I. Catalog only.
- **Notes:** Jansen discusses social media as alternative data. For FX specifically, the signal is much weaker than for equities. Not recommended as a primary data source.

### 3.6 Satellite / Transaction Data (Institutional)
- **URL:** Various (Orbital Insight, RS Metrics, Second Measure, Advan Research)
- **Data types:** Satellite imagery (parking lot counts, oil storage), credit card transactions, foot traffic, app usage
- **Cost tier:** Enterprise only — $50,000+/year typically
- **Historical coverage:** Varies. Satellite: ~5 years. Transaction: ~3-5 years.
- **API availability:** Varies by provider
- **Update frequency:** Daily to weekly
- **Quality assessment:** Documented alpha in academic literature. Used by large quant funds.
- **Project 2 relevance:** **LOW** — Far beyond budget. Cataloged per Jansen framework for completeness. Not actionable for this project.
- **Notes:** Pure catalog entry. Would require institutional research budget.

---

## Category 4: Crypto Data

### 4.1 Binance
- **URL:** https://binance-docs.github.io/apidocs/
- **Data types:** Crypto spot + futures. 600+ trading pairs. OHLCV candles, order book, trades, ticker, account data.
- **Cost tier:** Free API access (no account needed for market data)
- **Historical coverage:** From Binance launch (~2017) to present. M1 candles available. Older coins have deeper history.
- **API availability:** Yes — REST + WebSocket API. Rate limits: 1200 requests/min (weight-based). Python: `python-binance` library.
- **Update frequency:** Real-time streaming. Candles: 1m, 3m, 5m, 15m, 30m, 1h, 2h, 4h, 6h, 8h, 12h, 1d, 3d, 1w, 1M
- **Quality assessment:** Largest crypto exchange by volume. Data is genuine exchange data. Liquidity is highest for BTC/USDT, ETH/USDT. Some concerns about wash trading on smaller pairs.
- **Project 2 relevance:** **HIGH** (if crypto is in scope) — Best free source for crypto data. Rich API, deep history, high liquidity pairs. If Project 2 includes crypto as an asset class, Binance is the primary source.
- **Notes:** Availability depends on jurisdiction (Binance restricted in some countries). Binance.US has different API. Check local regulations.

### 4.2 CoinGecko
- **URL:** https://www.coingecko.com/en/api
- **Data types:** Crypto market data aggregated from 900+ exchanges. Price, volume, market cap, historical data, exchange data, derivatives, NFTs.
- **Cost tier:** Free tier: 30 calls/min. Pro: $129/month. Analyst: $499/month.
- **Historical coverage:** From coin inception for supported coins. Daily OHLCV going back years.
- **API availability:** Yes — REST API. Free tier is generous for research. Python: `pycoingecko` library.
- **Update frequency:** Every few minutes for prices; daily for historical OHLCV
- **Quality assessment:** Aggregated from multiple exchanges — more robust than single-exchange data. Widely used by crypto researchers. Free tier sufficient for daily+ analysis.
- **Project 2 relevance:** **MEDIUM** (if crypto in scope) — Good for daily+ crypto data and market overview. Binance better for intraday. CoinGecko better for cross-exchange aggregated prices.
- **Notes:** Rate limits on free tier may be constraining for large historical downloads. Pro plan removes most limits.

### 4.3 Coinbase (Advanced Trade API)
- **URL:** https://docs.cdp.coinbase.com/
- **Data types:** Crypto spot trading. BTC, ETH, and 200+ other crypto assets.
- **Cost tier:** Free API access with account
- **Historical coverage:** From Coinbase Pro launch (~2015 for BTC)
- **API availability:** Yes — REST + WebSocket. Rate limits apply.
- **Update frequency:** Real-time streaming
- **Quality assessment:** Major US-regulated exchange. Data quality is high. Lower pair count than Binance but fully regulated.
- **Project 2 relevance:** **LOW** (unless specifically needing US-regulated exchange data) — Binance + CoinGecko cover the crypto data needs. Coinbase useful only for US-specific regulatory compliance scenarios.
- **Notes:** Cataloged for completeness. Coinbase Pro data is available through multiple third-party data aggregators too.

### 4.4 Kaiko
- **URL:** https://www.kaiko.com/
- **Data types:** Institutional-grade crypto market data. Tick-by-tick trades, order book snapshots (L2/L3), OHLCV, VWAP, from 100+ exchanges.
- **Cost tier:** Enterprise — pricing not public (thousands/month)
- **Historical coverage:** From 2014 for major exchanges. Full order book history.
- **API availability:** REST + WebSocket + bulk downloads
- **Update frequency:** Real-time
- **Quality assessment:** Gold standard for institutional crypto data. Used by major quant funds and financial institutions.
- **Project 2 relevance:** **LOW** — Far exceeds budget. Cataloged for completeness.
- **Notes:** If Project 2 goes deeply into crypto microstructure, Kaiko is the reference. Not needed for standard OHLCV-level research.

---

## Category 5: Equity Indices

### 5.1 Yahoo Finance
- **URL:** https://finance.yahoo.com/
- **Data types:** US + global equities, indices (S&P 500, NASDAQ, DJI, Russell 2000), ETFs, mutual funds, options, FX, crypto. OHLCV, dividends, splits, fundamentals.
- **Cost tier:** Free (via `yfinance` Python library). No official API — `yfinance` reverse-engineers Yahoo's endpoints.
- **Historical coverage:** Daily: decades (back to 1920s for some indices). Intraday: 60 days at M1 resolution, 730 days at M5+.
- **API availability:** Unofficial — `yfinance` Python library (most popular, well-maintained). Subject to rate limiting and occasional breakage when Yahoo changes endpoints.
- **Update frequency:** ~15 min delayed for US equities; daily OHLCV within minutes of market close
- **Quality assessment:** Good enough for research. Adjusted close handles splits/dividends. Known issues: occasional data gaps, intraday history limited, corporate actions sometimes incorrect. Not suitable for production trading.
- **Project 2 relevance:** **MEDIUM** — Useful if equity indices are included as macro features for FX models (e.g., S&P 500 returns as a risk-on/risk-off signal). Free and easy to use.
- **Notes:** `yfinance` is the standard approach. Library is community-maintained, not officially supported by Yahoo. Breaks periodically when Yahoo changes backend.

### 5.2 Alpha Vantage (Equities & Indices)
- **URL:** https://www.alphavantage.co/documentation/
- **Data types:** US + global equities (daily OHLCV, adjusted), major indices (DJI, SPX, COMP, NDX, VIX, RUT — premium), fundamentals (income statement, balance sheet, cash flow, earnings), options, technical indicators
- **Cost tier:** Free (25 req/day); Premium plans for intraday, index data, real-time quotes. Basic premium: $49.99/month.
- **Historical coverage:** Daily equities: 20+ years with `outputsize=full`. Index data: depends on index.
- **API availability:** Yes — REST API. Equity: `TIME_SERIES_DAILY`, `TIME_SERIES_DAILY_ADJUSTED`, `TIME_SERIES_INTRADAY` (premium). Index: `INDEX_DATA` (premium). Fundamentals: `OVERVIEW`, `INCOME_STATEMENT`, `BALANCE_SHEET`, `CASH_FLOW`, `EARNINGS`. Also: `LISTING_STATUS`, `EARNINGS_CALENDAR`, `IPO_CALENDAR`.
- **Update frequency:** End-of-day for free; real-time with premium
- **Quality assessment:** Reliable for daily data. Fundamentals are GAAP/IFRS normalized. Index data requires premium. Free tier is very rate-limited.
- **Project 2 relevance:** **MEDIUM** — If equity indices are used as features, Alpha Vantage provides a single-API solution alongside FX and macro data. Premium required for index data.
- **Notes:** Index data (SPX, VIX, etc.) is premium. For FX-only Project 2, equity data is supplementary features only. VIX could be a high-value feature (volatility regime indicator).

### 5.3 Polygon.io
- **URL:** https://polygon.io/
- **Data types:** US equities, options, indices, FX, crypto. Tick-level data, aggregates, reference data.
- **Cost tier:** Free tier: 5 API calls/min, delayed data. Starter: $29/month. Developer: $79/month. Advanced: $199/month.
- **Historical coverage:** Equities: back to 2003. FX: back to 2018.
- **API availability:** Yes — REST + WebSocket API. Python: `polygon-api-client`. Well-documented.
- **Update frequency:** Real-time (paid tiers); 15-min delayed (free)
- **Quality assessment:** High quality. SIP data for US equities. Growing FX and crypto coverage. Used by fintech companies.
- **Project 2 relevance:** **LOW** — FX coverage starts only from 2018 (too short for 23-year backtests). Useful only if adding US equity data as features. OANDA + Dukascopy better for FX.
- **Notes:** FX historical depth is the limiting factor. Otherwise excellent API design.

---

## Category 6: Synthetic Data Approaches

### 6.1 TimeGAN (Time-series Generative Adversarial Network)
- **URL:** https://github.com/jsyoon0823/TimeGAN (reference implementation)
- **Data types:** Synthetic time series generation. Preserves temporal dynamics and statistical properties of real data.
- **Cost tier:** Free (open source)
- **Historical coverage:** N/A — generates synthetic data
- **API availability:** Python implementation (TensorFlow). Academic paper: Yoon, Jarrett, van der Schijf (NeurIPS 2019).
- **Update frequency:** N/A
- **Quality assessment:** State-of-the-art for time-series synthesis at time of publication. Jansen recommends in ML for Trading. Preserves autoregressive relationships. Quality depends on training data and hyperparameter tuning.
- **Project 2 relevance:** **HIGH** (for Part V) — Core method for Part V Synthetic Data Augmentation. Not needed in Part I but cataloged here for completeness.
- **Notes:** Reference implementation may need updating for current TensorFlow versions. Alternative implementations exist in PyTorch. Quality evaluation requires careful statistical testing (distributional tests, autocorrelation comparison, etc.).

### 6.2 Block Bootstrap Methods
- **URL:** N/A (statistical method, not a product)
- **Data types:** Resampled time series preserving local dependency structure
- **Cost tier:** Free (implement yourself)
- **Historical coverage:** N/A — generates synthetic data from real data
- **API availability:** `arch` Python package implements stationary and circular block bootstrap. `tsbootstrap` package also available.
- **Update frequency:** N/A
- **Quality assessment:** Well-established statistical method. Simpler than GANs. Preserves short-range dependencies within blocks. Loses long-range dependencies across blocks. Block size is a critical hyperparameter.
- **Project 2 relevance:** **MEDIUM** (for Part V) — Simpler baseline for data augmentation. Compare against TimeGAN to see if GAN complexity is justified.
- **Notes:** Variants: moving block bootstrap, circular block bootstrap, stationary bootstrap (Politis & Romano 1994). Block size selection: use the method of Politis & White (2004).

### 6.3 Other GAN Variants for Financial Time Series
- **URL:** Various academic papers
- **Data types:** Synthetic financial time series
- **Cost tier:** Free (academic implementations)
- **Historical coverage:** N/A
- **API availability:** Various GitHub implementations of varying quality
- **Update frequency:** N/A
- **Quality assessment:** Varies widely. Notable variants: RCGAN (Esteban et al.), SigWGAN (Ni et al. 2020), QuantGAN (Wiese et al. 2020). Active research area.
- **Project 2 relevance:** **SPECULATIVE** (for Part V) — Worth surveying in Part V literature review. TimeGAN is the baseline; these are alternatives if TimeGAN underperforms.
- **Notes:** SigWGAN uses signature features and Wasserstein distance — theoretically elegant. QuantGAN specifically targets financial return distributions. Both require significant implementation effort.

---

## Summary: Recommended Primary Sources for Project 2

| Category | Primary Source | Backup Source | Rationale |
|---|---|---|---|
| FX Price Data | OANDA (proven in P1) | Dukascopy (tick data) | Continuity + pipeline compatibility |
| Economic Calendar | Trading Economics API | FRED (via Alpha Vantage) | Calendar timestamps + surprise data |
| US Macro Data | FRED (via Alpha Vantage) | Trading Economics | Free, authoritative, easy API |
| Positioning Data | CFTC CoT (weekly) | — | Free, documented FX factor |
| Crypto (if in scope) | Binance | CoinGecko | Free, deepest liquidity |
| Equity Indices (features) | Yahoo Finance (`yfinance`) | Alpha Vantage | Free, sufficient for daily features |
| Commodities | Alpha Vantage | — | Free, covers gold/oil/copper |
| Synthetic Data (Part V) | TimeGAN | Block Bootstrap | Jansen-recommended + baseline |

### Budget Implications
- **Zero-cost stack:** OANDA (existing account) + FRED/Alpha Vantage (free) + HistData (free) + CoT (free) + Yahoo Finance (free) + Binance (free) + `pytrends` (free) = covers all primary needs
- **Low-cost additions:** HistData FTP ($7/mo), Alpha Vantage Premium ($49.99/mo), Trading Economics API (price TBD)
- **Not recommended:** TrueFX ($4,950/mo), Kaiko (enterprise), satellite/transaction data (enterprise)

### Data Quality Red Flags to Watch
1. **Look-ahead bias in macro data:** Use first-release values, not revised values, in backtests
2. **Survivorship bias:** Not relevant for FX pairs (they don't delist) but relevant if adding equities
3. **Time zone alignment:** Macro events in US time, FX data in GMT/UTC, broker data may be in broker timezone
4. **Vendor differences:** OANDA mid-price vs. Dukascopy bid/ask spread — ensure consistency
5. **Rate limit management:** Alpha Vantage (25/day free), Google Trends (undocumented), Binance (1200/min weighted)

---

## Open Questions for F-4 (Asset + Timeframe Selection)

1. **Scope decision:** Is crypto in scope for Project 2, or FX-only?
2. **Timeframe decision:** Will adaptive re-optimization operate at 4H (as P1) or explore other timeframes (daily, weekly)?
3. **Feature scope:** Will equity indices and commodities serve as features for FX models?
4. **Economic calendar:** Invest in Trading Economics API subscription, or build FRED-only pipeline?
5. **TrueFX free tier:** Is the historical tick data still available for free? Worth checking before finalizing tick-data source.
