SELECT 
    dcs.player_id,
    dcs.player_name, 
    dcs.position,
    dcs.height_w_shoes,
    dcs.weight,
    dcs.wingspan,
    dcs.standing_vertical_leap,
    dcs.max_vertical_leap,
    dcs.lane_agility_time,
    dcs.modified_lane_agility_time,
    dcs.three_quarter_sprint,
    dcs.bench_press,
    dcs.spot_fifteen_corner_left,
    dcs.spot_fifteen_break_left,
    dcs.spot_nba_corner_right,
    dcs.off_drib_college_top_key,
    dcs.hand_length,
    dcs.spot_nba_break_right,
    dcs.hand_width,
    dcs.spot_nba_top_key, 
    dcs.body_fat_pct, 
    dh.overall_pick AS draft_position,
    dh.round_number AS draft_round,
    dh.season AS draft_year
FROM 
    draft_combine_stats AS dcs
JOIN 
    draft_history AS dh
ON 
    dcs.player_id = dh.person_id;



