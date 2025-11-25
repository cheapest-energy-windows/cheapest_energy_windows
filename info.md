# Cheapest Energy Windows

## What is this?

This integration optimizes your energy consumption and battery storage by automatically identifying the cheapest charging windows and most expensive discharging periods based on dynamic electricity prices.

Perfect for users with:
- Variable electricity pricing (spot prices)
- Home battery systems
- Solar installations with feed-in tariffs
- Electric vehicle charging needs

## Key Features

- **Automatic Window Detection** - Finds optimal charge/discharge times
- **Statistical Analysis** - Uses percentile-based selection for accuracy
- **Battery Optimization** - Considers round-trip efficiency
- **Dual-Day Management** - Different settings for today/tomorrow
- **Time Overrides** - Force charging during specific periods
- **Full Dashboard** - Complete control interface included

## Quick Start

1. Install through HACS
2. Add integration in Settings
3. Follow guided setup wizard
4. Install dashboard via service call
5. Start saving on energy costs!

## Requirements

- Home Assistant 2024.1.0 or newer
- Electricity price sensor (Nordpool, ENTSO-E, Tibber, etc.)

## Support

- [Documentation](https://github.com/cheapest-energy-windows/cheapest_energy_windows)
- [Report Issues](https://github.com/cheapest-energy-windows/cheapest_energy_windows/issues)
- [Community Discussion](https://community.home-assistant.io/)

{% if installed %}
## Installed Version: {{ version }}

Thank you for using Cheapest Energy Windows!

### Quick Actions
- Call service `cheapest_energy_windows.install_dashboard` to install dashboard
- Check `sensor.cew_today` for current state
- Configure settings through the dashboard
{% endif %}