# Changelog

All notable changes to `cedardev-fixed-income` will be documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

## [1.0.0] — 2024-07-01

### Added
- `ust_pricer`: Full US Treasury pricing engine (ACT/ACT ICMA, YTM solver, duration, DV01, convexity, zero-curve bootstrap, Z-spread)
- `mbs_analytics`: Agency MBS analytics (PSA model, WAL, OAS, PSA sensitivity table)
- `treasury_futures`: CME Treasury futures (CTD, conversion factor, implied repo, gross/net basis)
- `var_engine`: VaR engine (Historical Simulation delta-gamma, Parametric EWMA, Expected Shortfall, Component VaR)
- `margin_engine`: SEC Rule 4210 margining pipeline (5-step: extract → categorize → net → cross-margin → validate)
- `model_validation`: Model validation suite (Kupiec, Christoffersen, Basel traffic light, FRTB PLA test)
