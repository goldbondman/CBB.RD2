import sys
from ingestion.market_lines import fetch_espn_scoreboard, fetch_pinnacle_lines, normalize_team_name, _extract_team_names_from_market, parse_espn_event
from datetime import date

espn = fetch_espn_scoreboard(date.today())
pinn = fetch_pinnacle_lines()

espn_teams = set()
for e in espn:
    p = parse_espn_event(e)
    if p:
        espn_teams.add(normalize_team_name(p['home_team_name']))
        espn_teams.add(normalize_team_name(p['away_team_name']))

pinn_teams = set()
for m in pinn:
    h, a = _extract_team_names_from_market(m)
    pinn_teams.add(normalize_team_name(h))
    pinn_teams.add(normalize_team_name(a))

print("ESPN teams:", list(espn_teams)[:10])
print("Pinnacle teams:", list(pinn_teams)[:10])
print("Intersection:", len(espn_teams.intersection(pinn_teams)))
print("ESPN total:", len(espn_teams))
print("Pinn total:", len(pinn_teams))
