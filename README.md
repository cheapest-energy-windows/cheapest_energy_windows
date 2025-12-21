# Cheapest Energy Windows for Home Assistant

[![HACS](https://my.home-assistant.io/badges/hacs_repository.svg)](https://my.home-assistant.io/redirect/hacs_repository/?owner=cheapest-energy-windows&repository=cheapest_energy_windows&category=integration)
[![GitHub Release](https://img.shields.io/github/release/cheapest-energy-windows/cheapest_energy_windows.svg)](https://github.com/cheapest-energy-windows/cheapest_energy_windows/releases)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A Home Assistant integration that helps you find the best times to charge and discharge your battery based on dynamic electricity prices. It simulates your daily energy costs and identifies profitable arbitrage windows.

## What It Does

- Analyzes hourly/quarter-hourly electricity prices from Nord Pool or ENTSO-E
- Identifies the cheapest windows for charging and most expensive for discharging
- Calculates if the price spread is actually profitable after battery losses
- Simulates expected costs and savings for the day
- Provides sensors that trigger your battery automations at the right times

You configure your battery specs (capacity, charge/discharge power, efficiency), plug in your price sensor, and the integration tells you when to charge/discharge. Connect your existing battery automations to the sensor states - when `sensor.cew_today` changes to `charge`, your battery charges.

## Dashboard

![Dashboard Preview](CEW-Dashboard.jpg?v=2)

The dashboard shows today's and tomorrow's price charts, which windows are selected, current battery state, and estimated costs. Available as a [separate HACS package](https://github.com/cheapest-energy-windows/cheapest_energy_windows_dashboard).

## Quick Start

1. Install via HACS (search "Cheapest Energy Windows")
2. Add the integration and select your price sensor (Nord Pool or ENTSO-E)
3. Configure your battery parameters
4. Install the [dashboard package](https://github.com/cheapest-energy-windows/cheapest_energy_windows_dashboard)
5. Connect your battery automations to the `sensor.cew_today` states

## Supported Price Sources

- **Nord Pool** - Nordic and Baltic countries
- **ENTSO-E** - European electricity market

Prices must be in EUR/kWh. Both 15-minute and 1-hour window modes are supported.

## How Window Selection Works

1. Prices are sorted and filtered by percentile (e.g., cheapest 25%)
2. Windows are selected only if the spread between cheap and expensive prices covers battery round-trip losses
3. The algorithm accounts for base household usage during charge/discharge
4. Results are a list of timestamps when charging or discharging is profitable

## Key Settings

| Setting | Description |
|---------|-------------|
| Charging Windows | Max number of charge windows to select |
| Discharge Windows | Max number of discharge windows to select |
| Percentile Threshold | Only consider prices in the cheapest/most expensive X% |
| Min Profit Threshold | Required profit margin after RTE losses |
| Battery RTE | Round-trip efficiency (typically 85-90%) |
| Charge/Discharge Power | Your battery's power limits in Watts |

## Sensor States

`sensor.cew_today` outputs one of:
- `charge` - Currently in a cheap window, charging is profitable
- `discharge` - Currently in an expensive window, discharging is profitable
- `idle` - No action recommended
- `off` - Automation disabled

Trigger your battery control automations on these state changes.

## Requirements

- Home Assistant 2024.1+
- A dynamic price sensor (Nord Pool or ENTSO-E integration)
- Dashboard requires: Mushroom Cards, ApexCharts Card, Fold Entity Row, Card Mod (all via HACS)

## Documentation

For detailed configuration, automation examples, and troubleshooting, see the [Wiki](https://github.com/cheapest-energy-windows/cheapest_energy_windows/wiki).

## License

MIT License
