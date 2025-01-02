SELECT 
    gi.game_id,
    gi.game_date,
    gi.attendance,
    g.team_abbreviation_home AS home_team,
    g.team_abbreviation_away AS away_team
FROM 
    game_info gi
JOIN 
    game g ON gi.game_id = g.game_id
ORDER BY 
    gi.attendance DESC

    
    
SELECT 
    gi.game_id,
    gi.game_date,
    gi.attendance,
    g.team_abbreviation_home AS home_team,
    g.team_abbreviation_away AS away_team
FROM 
    game_info gi
JOIN 
    game g ON gi.game_id = g.game_id
WHERE 
    (g.team_abbreviation_home = 'WST' AND g.team_abbreviation_away = 'EST')
    OR 
    (g.team_abbreviation_home = 'EST' AND g.team_abbreviation_away = 'WST')
ORDER BY 
    gi.attendance DESC



