# Claude Code Project Context

## Home Assistant Dev Environment

### API Access
- **URL:** `http://10.50.0.9:8125`
- **Token:** `eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiI1NzhmOWE3NjE0MDI0ODQ3OGE4Nzc4ZWU3NjhkYjhiZiIsImlhdCI6MTc2NjQzMjQ2NSwiZXhwIjoyMDgxNzkyNDY1fQ.DM725h43dPRb_bOIYXTX7TZwYeBv-xwLoC_B853tfLs`

### Useful API Commands

Query CEW today sensor:
```bash
curl -s "http://10.50.0.9:8125/api/states/sensor.cew_today" -H "Authorization: Bearer <token>" | python3 -m json.tool
```

Query any entity:
```bash
curl -s "http://10.50.0.9:8125/api/states/<entity_id>" -H "Authorization: Bearer <token>"
```

List all CEW entities:
```bash
curl -s "http://10.50.0.9:8125/api/states" -H "Authorization: Bearer <token>" | python3 -c "import sys,json; [print(e['entity_id']) for e in json.load(sys.stdin) if 'cew_' in e['entity_id']]"
```

Reload integration:
```bash
curl -X POST "http://10.50.0.9:8125/api/services/homeassistant/reload_config_entry" -H "Authorization: Bearer <token>" -H "Content-Type: application/json" -d '{"entry_id": "<config_entry_id>"}'
```

## File Sync

Sync changes to dev environment:
```bash
rsync -av /Users/antonio/Documents/Github/cheapest_energy_windows/custom_components/cheapest_energy_windows/ /Volumes/docker/homeassistant_dev/custom_components/cheapest_energy_windows/
```

## Key Files

- `calculation_engine.py` - Core price/window calculations
- `coordinator.py` - Data update coordinator
- `sensor.py` - CEW Today/Tomorrow sensors
- `__init__.py` - Integration setup/teardown
- `const.py` - Constants and CALCULATION_AFFECTING_KEYS

## Entity Naming

- Prefix: `cew_`
- Examples: `sensor.cew_today`, `switch.cew_automation_enabled`, `number.cew_charge_power`
