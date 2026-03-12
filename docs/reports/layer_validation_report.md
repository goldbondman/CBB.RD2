# Layer Validation Report

Generated at UTC: 2026-03-12T18:00:37Z
Total rows: 335

## Statistically Supported

```text
             layer_name         scenario market  n  hit_rate     lift  p_value                     notes
one_possession_seed_gap upset_validation     ml 60      0.45 0.165481  0.00462 baseline_upset_rate=0.285
```

## Weak / Inconclusive

```text
                                           layer_name                  scenario        market   n  hit_rate     lift  p_value                                   notes
                                both_teams_fast_tempo          situational_over         total   0       NaN      NaN      NaN                   ⛔ INSUFFICIENT SAMPLE
                                   pace_mismatch_over          situational_over         total   0       NaN      NaN      NaN                   ⛔ INSUFFICIENT SAMPLE
                                   neither_team_slows          situational_over         total   0       NaN      NaN      NaN                   ⛔ INSUFFICIENT SAMPLE
                                   low_foul_rate_both          situational_over         total   0       NaN      NaN      NaN                   ⛔ INSUFFICIENT SAMPLE
                                   both_high_live_tov          situational_over         total   0       NaN      NaN      NaN                   ⛔ INSUFFICIENT SAMPLE
                                    both_efg_above_52          situational_over         total   0       NaN      NaN      NaN                   ⛔ INSUFFICIENT SAMPLE
                               high_combined_3pa_rate          situational_over         total   0       NaN      NaN      NaN                   ⛔ INSUFFICIENT SAMPLE
                                   both_poor_defenses          situational_over         total   0       NaN      NaN      NaN                   ⛔ INSUFFICIENT SAMPLE
                                     low_ft_rate_both          situational_over         total   0       NaN      NaN      NaN                   ⛔ INSUFFICIENT SAMPLE
                             neither_elite_interior_d          situational_over         total   0       NaN      NaN      NaN                   ⛔ INSUFFICIENT SAMPLE
                                 both_mti_trending_up          situational_over         total   0       NaN      NaN      NaN                   ⛔ INSUFFICIENT SAMPLE
                                    high_combined_spr          situational_over         total   0       NaN      NaN      NaN                   ⛔ INSUFFICIENT SAMPLE
                                   both_odi_offensive          situational_over         total   0       NaN      NaN      NaN                   ⛔ INSUFFICIENT SAMPLE
                                             high_pmi          situational_over         total   0       NaN      NaN      NaN                   ⛔ INSUFFICIENT SAMPLE
                                         both_low_dpc          situational_over         total   0       NaN      NaN      NaN                   ⛔ INSUFFICIENT SAMPLE
                         first_meeting_different_conf          situational_over         total   0       NaN      NaN      NaN                   ⛔ INSUFFICIENT SAMPLE
                                   both_coming_off_ot          situational_over         total   0       NaN      NaN      NaN                   ⛔ INSUFFICIENT SAMPLE
                                   early_conf_tourney          situational_over         total   0       NaN      NaN      NaN                   ⛔ INSUFFICIENT SAMPLE
                                  market_anchored_low          situational_over         total   0       NaN      NaN      NaN                   ⛔ INSUFFICIENT SAMPLE
                                     elimination_game      situational_underdog           ats  76  0.526316 0.050509 0.144865                             🔍 PROMISING
                                  blue_blood_opponent      situational_underdog           ats   0       NaN      NaN      NaN                   ⛔ INSUFFICIENT SAMPLE
                                  blue_blood_opponent      situational_underdog            ml   0       NaN      NaN      NaN                   ⛔ INSUFFICIENT SAMPLE
                               opponent_top_25_ranked      situational_underdog           ats   0       NaN      NaN      NaN                   ⛔ INSUFFICIENT SAMPLE
                               opponent_top_25_ranked      situational_underdog            ml   0       NaN      NaN      NaN                   ⛔ INSUFFICIENT SAMPLE
                               opponent_on_win_streak      situational_underdog           ats   0       NaN      NaN      NaN                   ⛔ INSUFFICIENT SAMPLE
                               opponent_on_win_streak      situational_underdog            ml   0       NaN      NaN      NaN                   ⛔ INSUFFICIENT SAMPLE
                               underdog_just_lost_big      situational_underdog           ats   0       NaN      NaN      NaN                   ⛔ INSUFFICIENT SAMPLE
                               underdog_just_lost_big      situational_underdog            ml   0       NaN      NaN      NaN                   ⛔ INSUFFICIENT SAMPLE
                        efficiency_closer_than_spread      situational_underdog           ats   0       NaN      NaN      NaN                   ⛔ INSUFFICIENT SAMPLE
                        efficiency_closer_than_spread      situational_underdog            ml   0       NaN      NaN      NaN                   ⛔ INSUFFICIENT SAMPLE
                             underdog_losses_vs_top25      situational_underdog           ats   0       NaN      NaN      NaN                   ⛔ INSUFFICIENT SAMPLE
                             underdog_losses_vs_top25      situational_underdog            ml   0       NaN      NaN      NaN                   ⛔ INSUFFICIENT SAMPLE
                               favorite_soft_schedule      situational_underdog           ats   0       NaN      NaN      NaN                   ⛔ INSUFFICIENT SAMPLE
                               favorite_soft_schedule      situational_underdog            ml   0       NaN      NaN      NaN                   ⛔ INSUFFICIENT SAMPLE
                                        pei_gap_small      situational_underdog           ats   0       NaN      NaN      NaN                   ⛔ INSUFFICIENT SAMPLE
                                        pei_gap_small      situational_underdog            ml   0       NaN      NaN      NaN                   ⛔ INSUFFICIENT SAMPLE
                           underdog_oreb_vs_poor_dreb      situational_underdog           ats   0       NaN      NaN      NaN                   ⛔ INSUFFICIENT SAMPLE
                           underdog_oreb_vs_poor_dreb      situational_underdog            ml   0       NaN      NaN      NaN                   ⛔ INSUFFICIENT SAMPLE
                     underdog_fast_vs_transition_weak      situational_underdog           ats   0       NaN      NaN      NaN                   ⛔ INSUFFICIENT SAMPLE
                     underdog_fast_vs_transition_weak      situational_underdog            ml   0       NaN      NaN      NaN                   ⛔ INSUFFICIENT SAMPLE
                       underdog_3pt_vs_poor_perimeter      situational_underdog           ats   0       NaN      NaN      NaN                   ⛔ INSUFFICIENT SAMPLE
                       underdog_3pt_vs_poor_perimeter      situational_underdog            ml   0       NaN      NaN      NaN                   ⛔ INSUFFICIENT SAMPLE
                       underdog_low_spr_favorite_high      situational_underdog           ats   0       NaN      NaN      NaN                   ⛔ INSUFFICIENT SAMPLE
                       underdog_low_spr_favorite_high      situational_underdog            ml   0       NaN      NaN      NaN                   ⛔ INSUFFICIENT SAMPLE
                    rested_underdog_fatigued_favorite      situational_underdog           ats   0       NaN      NaN      NaN                   ⛔ INSUFFICIENT SAMPLE
                    rested_underdog_fatigued_favorite      situational_underdog            ml   0       NaN      NaN      NaN                   ⛔ INSUFFICIENT SAMPLE
                                neutral_site_underdog      situational_underdog           ats   0       NaN      NaN      NaN                   ⛔ INSUFFICIENT SAMPLE
                                neutral_site_underdog      situational_underdog            ml   0       NaN      NaN      NaN                   ⛔ INSUFFICIENT SAMPLE
                                         revenge_spot      situational_underdog           ats   0       NaN      NaN      NaN                   ⛔ INSUFFICIENT SAMPLE
                                         revenge_spot      situational_underdog            ml   0       NaN      NaN      NaN                   ⛔ INSUFFICIENT SAMPLE
                               third_meeting_underdog      situational_underdog           ats   0       NaN      NaN      NaN                   ⛔ INSUFFICIENT SAMPLE
                               third_meeting_underdog      situational_underdog            ml   0       NaN      NaN      NaN                   ⛔ INSUFFICIENT SAMPLE
                                reverse_line_movement      situational_underdog           ats   0       NaN      NaN      NaN                   ⛔ INSUFFICIENT SAMPLE
                                reverse_line_movement      situational_underdog            ml   0       NaN      NaN      NaN                   ⛔ INSUFFICIENT SAMPLE
                        underdog_mti_up_favorite_flat      situational_underdog           ats   0       NaN      NaN      NaN                   ⛔ INSUFFICIENT SAMPLE
                        underdog_mti_up_favorite_flat      situational_underdog            ml   0       NaN      NaN      NaN                   ⛔ INSUFFICIENT SAMPLE
                           underdog_sci_battle_tested      situational_underdog           ats   0       NaN      NaN      NaN                   ⛔ INSUFFICIENT SAMPLE
                           underdog_sci_battle_tested      situational_underdog            ml   0       NaN      NaN      NaN                   ⛔ INSUFFICIENT SAMPLE
                               underdog_odi_defensive      situational_underdog           ats   0       NaN      NaN      NaN                   ⛔ INSUFFICIENT SAMPLE
                               underdog_odi_defensive      situational_underdog            ml   0       NaN      NaN      NaN                   ⛔ INSUFFICIENT SAMPLE
                                        large_ane_gap   situational_blowout_win           ats   0       NaN      NaN      NaN                   ⛔ INSUFFICIENT SAMPLE
                 both_top50_off_opponent_bottom50_def   situational_blowout_win           ats   0       NaN      NaN      NaN                   ⛔ INSUFFICIENT SAMPLE
                                      pei_gap_extreme   situational_blowout_win           ats   0       NaN      NaN      NaN                   ⛔ INSUFFICIENT SAMPLE
                              opponent_short_rotation   situational_blowout_win           ats   0       NaN      NaN      NaN                   ⛔ INSUFFICIENT SAMPLE
                                      depth_advantage   situational_blowout_win           ats   0       NaN      NaN      NaN                   ⛔ INSUFFICIENT SAMPLE
                   team_fast_opponent_transition_weak   situational_blowout_win           ats   0       NaN      NaN      NaN                   ⛔ INSUFFICIENT SAMPLE
            team_elite_ft_drawing_opponent_foul_prone   situational_blowout_win           ats   0       NaN      NaN      NaN                   ⛔ INSUFFICIENT SAMPLE
                    team_high_oreb_opponent_poor_dreb   situational_blowout_win           ats   0       NaN      NaN      NaN                   ⛔ INSUFFICIENT SAMPLE
       team_top25_fastbreak_opponent_bottom25_allowed   situational_blowout_win           ats   0       NaN      NaN      NaN                   ⛔ INSUFFICIENT SAMPLE
              opponent_high_tov_team_forces_turnovers   situational_blowout_win           ats   0       NaN      NaN      NaN                   ⛔ INSUFFICIENT SAMPLE
                    opponent_poor_ft_team_draws_fouls   situational_blowout_win           ats   0       NaN      NaN      NaN                   ⛔ INSUFFICIENT SAMPLE
                   opponent_short_rotation_fast_tempo   situational_blowout_win           ats   0       NaN      NaN      NaN                   ⛔ INSUFFICIENT SAMPLE
                    opponent_bottom25_second_half_def   situational_blowout_win           ats   0       NaN      NaN      NaN                   ⛔ INSUFFICIENT SAMPLE
                   rested_team_opponent_game3_in3days   situational_blowout_win           ats   0       NaN      NaN      NaN                   ⛔ INSUFFICIENT SAMPLE
                                opponent_prev_game_ot   situational_blowout_win           ats   0       NaN      NaN      NaN                   ⛔ INSUFFICIENT SAMPLE
                  no_tournament_implications_opponent   situational_blowout_win           ats   0       NaN      NaN      NaN                   ⛔ INSUFFICIENT SAMPLE
                         opponent_pseudo_road_neutral   situational_blowout_win           ats   0       NaN      NaN      NaN                   ⛔ INSUFFICIENT SAMPLE
                                        mti_crossover   situational_blowout_win           ats   0       NaN      NaN      NaN                   ⛔ INSUFFICIENT SAMPLE
                        sci_battle_tested_vs_untested   situational_blowout_win           ats   0       NaN      NaN      NaN                   ⛔ INSUFFICIENT SAMPLE
                                 odi_extreme_mismatch   situational_blowout_win           ats   0       NaN      NaN      NaN                   ⛔ INSUFFICIENT SAMPLE
                                    ane_gap_plus_rest   situational_blowout_win           ats   0       NaN      NaN      NaN                   ⛔ INSUFFICIENT SAMPLE
                               efficiency_gap_under_8   situational_underdog_ml            ml   0       NaN      NaN      NaN                   ⛔ INSUFFICIENT SAMPLE
                         both_within_8_adj_efficiency   situational_underdog_ml            ml   0       NaN      NaN      NaN                   ⛔ INSUFFICIENT SAMPLE
                  underdog_top50_adj_eff_despite_odds   situational_underdog_ml            ml   0       NaN      NaN      NaN                   ⛔ INSUFFICIENT SAMPLE
                                      pei_gap_under_5   situational_underdog_ml            ml   0       NaN      NaN      NaN                   ⛔ INSUFFICIENT SAMPLE
   underdog_elite_def_vs_favorite_halfcourt_struggles   situational_underdog_ml            ml   0       NaN      NaN      NaN                   ⛔ INSUFFICIENT SAMPLE
                        underdog_slow_tempo_elite_def   situational_underdog_ml            ml   0       NaN      NaN      NaN                   ⛔ INSUFFICIENT SAMPLE
                 underdog_3pt_heavy_vs_poor_perimeter   situational_underdog_ml            ml   0       NaN      NaN      NaN                   ⛔ INSUFFICIENT SAMPLE
           underdog_superior_ft_drawing_vs_foul_prone   situational_underdog_ml            ml   0       NaN      NaN      NaN                   ⛔ INSUFFICIENT SAMPLE
                     underdog_top25_oreb_vs_poor_dreb   situational_underdog_ml            ml   0       NaN      NaN      NaN                   ⛔ INSUFFICIENT SAMPLE
                  efficiency_gap_under_10_elimination   situational_underdog_ml            ml   0       NaN      NaN      NaN                   ⛔ INSUFFICIENT SAMPLE
                underdog_tourney_experience_advantage   situational_underdog_ml            ml   0       NaN      NaN      NaN                   ⛔ INSUFFICIENT SAMPLE
               underdog_physical_conf_vs_finesse_conf   situational_underdog_ml            ml   0       NaN      NaN      NaN                   ⛔ INSUFFICIENT SAMPLE
underdog_elite_perimeter_scorer_vs_poor_perimeter_def   situational_underdog_ml            ml   0       NaN      NaN      NaN                   ⛔ INSUFFICIENT SAMPLE
                        revenge_game_large_prior_loss   situational_underdog_ml            ml   0       NaN      NaN      NaN                   ⛔ INSUFFICIENT SAMPLE
                                letdown_spot_favorite   situational_underdog_ml            ml   0       NaN      NaN      NaN                   ⛔ INSUFFICIENT SAMPLE
                    rested_underdog_fatigued_favorite   situational_underdog_ml            ml   0       NaN      NaN      NaN                   ⛔ INSUFFICIENT SAMPLE
                                neutral_site_underdog   situational_underdog_ml            ml   0       NaN      NaN      NaN                   ⛔ INSUFFICIENT SAMPLE
                        public_80pct_plus_on_favorite   situational_underdog_ml            ml   0       NaN      NaN      NaN                   ⛔ INSUFFICIENT SAMPLE
                       reverse_line_movement_underdog   situational_underdog_ml            ml   0       NaN      NaN      NaN                   ⛔ INSUFFICIENT SAMPLE
                               mti_momentum_crossover   situational_underdog_ml            ml   0       NaN      NaN      NaN                   ⛔ INSUFFICIENT SAMPLE
        underdog_odi_defensive_favorite_odi_offensive   situational_underdog_ml            ml   0       NaN      NaN      NaN                   ⛔ INSUFFICIENT SAMPLE
                        underdog_sci_highest_quartile   situational_underdog_ml            ml   0       NaN      NaN      NaN                   ⛔ INSUFFICIENT SAMPLE
                   low_spr_underdog_high_spr_favorite   situational_underdog_ml            ml   0       NaN      NaN      NaN                   ⛔ INSUFFICIENT SAMPLE
                               both_teams_top50_tempo  situational_blowout_over         total   0       NaN      NaN      NaN                   ⛔ INSUFFICIENT SAMPLE
                      neither_team_defensive_identity  situational_blowout_over         total   0       NaN      NaN      NaN                   ⛔ INSUFFICIENT SAMPLE
                           both_teams_top25_fastbreak  situational_blowout_over         total   0       NaN      NaN      NaN                   ⛔ INSUFFICIENT SAMPLE
                              both_low_halfcourt_rate  situational_blowout_over         total   0       NaN      NaN      NaN                   ⛔ INSUFFICIENT SAMPLE
                  combined_possessions_exceed_implied  situational_blowout_over         total   0       NaN      NaN      NaN                   ⛔ INSUFFICIENT SAMPLE
                             both_def_rating_bottom50  situational_blowout_over         total   0       NaN      NaN      NaN                   ⛔ INSUFFICIENT SAMPLE
                            both_efg_allowed_above_54  situational_blowout_over         total   0       NaN      NaN      NaN                   ⛔ INSUFFICIENT SAMPLE
                          both_poor_perimeter_defense  situational_blowout_over         total   0       NaN      NaN      NaN                   ⛔ INSUFFICIENT SAMPLE
                           both_poor_interior_defense  situational_blowout_over         total   0       NaN      NaN      NaN                   ⛔ INSUFFICIENT SAMPLE
                        neither_team_forces_turnovers  situational_blowout_over         total   0       NaN      NaN      NaN                   ⛔ INSUFFICIENT SAMPLE
                                    both_efg_above_54  situational_blowout_over         total   0       NaN      NaN      NaN                   ⛔ INSUFFICIENT SAMPLE
                            four_way_3pt_confirmation  situational_blowout_over         total   0       NaN      NaN      NaN                   ⛔ INSUFFICIENT SAMPLE
                                     both_low_ft_rate  situational_blowout_over         total   0       NaN      NaN      NaN                   ⛔ INSUFFICIENT SAMPLE
                             both_ppp_above_threshold  situational_blowout_over         total   0       NaN      NaN      NaN                   ⛔ INSUFFICIENT SAMPLE
                         first_meeting_different_conf  situational_blowout_over         total   0       NaN      NaN      NaN                   ⛔ INSUFFICIENT SAMPLE
                    both_coming_off_low_scoring_games  situational_blowout_over         total   0       NaN      NaN      NaN                   ⛔ INSUFFICIENT SAMPLE
                               early_tournament_round  situational_blowout_over         total   0       NaN      NaN      NaN                   ⛔ INSUFFICIENT SAMPLE
                         neither_team_true_home_court  situational_blowout_over         total   0       NaN      NaN      NaN                   ⛔ INSUFFICIENT SAMPLE
                                        both_high_spr  situational_blowout_over         total   0       NaN      NaN      NaN                   ⛔ INSUFFICIENT SAMPLE
                                   both_odi_offensive  situational_blowout_over         total   0       NaN      NaN      NaN                   ⛔ INSUFFICIENT SAMPLE
                                 both_mti_trending_up  situational_blowout_over         total   0       NaN      NaN      NaN                   ⛔ INSUFFICIENT SAMPLE
                                   high_pmi_both_fast  situational_blowout_over         total   0       NaN      NaN      NaN                   ⛔ INSUFFICIENT SAMPLE
                                  both_bottom25_tempo situational_blowout_under         total   0       NaN      NaN      NaN                   ⛔ INSUFFICIENT SAMPLE
                           both_top25_force_slow_pace situational_blowout_under         total   0       NaN      NaN      NaN                   ⛔ INSUFFICIENT SAMPLE
                   combined_possessions_below_implied situational_blowout_under         total   0       NaN      NaN      NaN                   ⛔ INSUFFICIENT SAMPLE
                             both_high_halfcourt_rate situational_blowout_under         total   0       NaN      NaN      NaN                   ⛔ INSUFFICIENT SAMPLE
                                  both_high_foul_rate situational_blowout_under         total   0       NaN      NaN      NaN                   ⛔ INSUFFICIENT SAMPLE
                        both_top25_adj_def_efficiency situational_blowout_under         total   0       NaN      NaN      NaN                   ⛔ INSUFFICIENT SAMPLE
                               both_top25_efg_allowed situational_blowout_under         total   0       NaN      NaN      NaN                   ⛔ INSUFFICIENT SAMPLE
                                  both_force_high_tov situational_blowout_under         total   0       NaN      NaN      NaN                   ⛔ INSUFFICIENT SAMPLE
                                both_top25_block_rate situational_blowout_under         total   0       NaN      NaN      NaN                   ⛔ INSUFFICIENT SAMPLE
                      both_top25_opponent_3pt_allowed situational_blowout_under         total   0       NaN      NaN      NaN                   ⛔ INSUFFICIENT SAMPLE
                                    both_efg_below_48 situational_blowout_under         total   0       NaN      NaN      NaN                   ⛔ INSUFFICIENT SAMPLE
                                    both_low_3pa_rate situational_blowout_under         total   0       NaN      NaN      NaN                   ⛔ INSUFFICIENT SAMPLE
                                       both_poor_oreb situational_blowout_under         total   0       NaN      NaN      NaN                   ⛔ INSUFFICIENT SAMPLE
                              both_high_ft_dependency situational_blowout_under         total   0       NaN      NaN      NaN                   ⛔ INSUFFICIENT SAMPLE
                              third_meeting_same_conf situational_blowout_under         total   0       NaN      NaN      NaN                   ⛔ INSUFFICIENT SAMPLE
                   both_coming_off_high_scoring_games situational_blowout_under         total   0       NaN      NaN      NaN                   ⛔ INSUFFICIENT SAMPLE
                         high_stakes_elimination_both situational_blowout_under         total   0       NaN      NaN      NaN                   ⛔ INSUFFICIENT SAMPLE
                                back_to_back_one_team situational_blowout_under         total   0       NaN      NaN      NaN                   ⛔ INSUFFICIENT SAMPLE
                                         both_low_spr situational_blowout_under         total   0       NaN      NaN      NaN                   ⛔ INSUFFICIENT SAMPLE
                                   both_odi_defensive situational_blowout_under         total   0       NaN      NaN      NaN                   ⛔ INSUFFICIENT SAMPLE
                           both_mti_flat_or_declining situational_blowout_under         total   0       NaN      NaN      NaN                   ⛔ INSUFFICIENT SAMPLE
                                     low_combined_pmi situational_blowout_under         total   0       NaN      NaN      NaN                   ⛔ INSUFFICIENT SAMPLE
                                  five_twelve_matchup     march_madness_rd1_rd2           ats   0       NaN      NaN      NaN                   ⛔ INSUFFICIENT SAMPLE
                                  five_twelve_matchup     march_madness_rd1_rd2            ml   0       NaN      NaN      NaN                   ⛔ INSUFFICIENT SAMPLE
                                  five_twelve_matchup     march_madness_rd1_rd2         total   0       NaN      NaN      NaN                   ⛔ INSUFFICIENT SAMPLE
                                   six_eleven_matchup     march_madness_rd1_rd2           ats   0       NaN      NaN      NaN                   ⛔ INSUFFICIENT SAMPLE
                                   six_eleven_matchup     march_madness_rd1_rd2            ml   0       NaN      NaN      NaN                   ⛔ INSUFFICIENT SAMPLE
                                   six_eleven_matchup     march_madness_rd1_rd2         total   0       NaN      NaN      NaN                   ⛔ INSUFFICIENT SAMPLE
                                   eight_nine_matchup     march_madness_rd1_rd2           ats   0       NaN      NaN      NaN                   ⛔ INSUFFICIENT SAMPLE
                                   eight_nine_matchup     march_madness_rd1_rd2            ml   0       NaN      NaN      NaN                   ⛔ INSUFFICIENT SAMPLE
                                   eight_nine_matchup     march_madness_rd1_rd2         total   0       NaN      NaN      NaN                   ⛔ INSUFFICIENT SAMPLE
                    double_digit_seed_high_efficiency     march_madness_rd1_rd2           ats   0       NaN      NaN      NaN                   ⛔ INSUFFICIENT SAMPLE
                    double_digit_seed_high_efficiency     march_madness_rd1_rd2            ml   0       NaN      NaN      NaN                   ⛔ INSUFFICIENT SAMPLE
                    double_digit_seed_high_efficiency     march_madness_rd1_rd2         total   0       NaN      NaN      NaN                   ⛔ INSUFFICIENT SAMPLE
                         blue_blood_favorite_inflated     march_madness_rd1_rd2           ats   0       NaN      NaN      NaN                   ⛔ INSUFFICIENT SAMPLE
                         blue_blood_favorite_inflated     march_madness_rd1_rd2            ml   0       NaN      NaN      NaN                   ⛔ INSUFFICIENT SAMPLE
                         blue_blood_favorite_inflated     march_madness_rd1_rd2         total   0       NaN      NaN      NaN                   ⛔ INSUFFICIENT SAMPLE
                        seed_diff_misleads_efficiency     march_madness_rd1_rd2           ats   0       NaN      NaN      NaN                   ⛔ INSUFFICIENT SAMPLE
                        seed_diff_misleads_efficiency     march_madness_rd1_rd2            ml   0       NaN      NaN      NaN                   ⛔ INSUFFICIENT SAMPLE
                        seed_diff_misleads_efficiency     march_madness_rd1_rd2         total   0       NaN      NaN      NaN                   ⛔ INSUFFICIENT SAMPLE
                 mid_major_top100_efficiency_underdog     march_madness_rd1_rd2           ats   0       NaN      NaN      NaN                   ⛔ INSUFFICIENT SAMPLE
                 mid_major_top100_efficiency_underdog     march_madness_rd1_rd2            ml   0       NaN      NaN      NaN                   ⛔ INSUFFICIENT SAMPLE
                 mid_major_top100_efficiency_underdog     march_madness_rd1_rd2         total   0       NaN      NaN      NaN                   ⛔ INSUFFICIENT SAMPLE
                                    rd1_bye_advantage     march_madness_rd1_rd2           ats   0       NaN      NaN      NaN                   ⛔ INSUFFICIENT SAMPLE
                                    rd1_bye_advantage     march_madness_rd1_rd2            ml   0       NaN      NaN      NaN                   ⛔ INSUFFICIENT SAMPLE
                                    rd1_bye_advantage     march_madness_rd1_rd2         total   0       NaN      NaN      NaN                   ⛔ INSUFFICIENT SAMPLE
                  opponent_conf_tourney_champ_fatigue     march_madness_rd1_rd2           ats   0       NaN      NaN      NaN                   ⛔ INSUFFICIENT SAMPLE
                  opponent_conf_tourney_champ_fatigue     march_madness_rd1_rd2            ml   0       NaN      NaN      NaN                   ⛔ INSUFFICIENT SAMPLE
                  opponent_conf_tourney_champ_fatigue     march_madness_rd1_rd2         total   0       NaN      NaN      NaN                   ⛔ INSUFFICIENT SAMPLE
                         rest_differential_2plus_days     march_madness_rd1_rd2           ats   0       NaN      NaN      NaN                   ⛔ INSUFFICIENT SAMPLE
                         rest_differential_2plus_days     march_madness_rd1_rd2            ml   0       NaN      NaN      NaN                   ⛔ INSUFFICIENT SAMPLE
                         rest_differential_2plus_days     march_madness_rd1_rd2         total   0       NaN      NaN      NaN                   ⛔ INSUFFICIENT SAMPLE
              opponent_conf_tourney_runner_up_fatigue     march_madness_rd1_rd2           ats   0       NaN      NaN      NaN                   ⛔ INSUFFICIENT SAMPLE
              opponent_conf_tourney_runner_up_fatigue     march_madness_rd1_rd2            ml   0       NaN      NaN      NaN                   ⛔ INSUFFICIENT SAMPLE
              opponent_conf_tourney_runner_up_fatigue     march_madness_rd1_rd2         total   0       NaN      NaN      NaN                   ⛔ INSUFFICIENT SAMPLE
                                 pseudo_home_pod_game     march_madness_rd1_rd2           ats   0       NaN      NaN      NaN                   ⛔ INSUFFICIENT SAMPLE
                                 pseudo_home_pod_game     march_madness_rd1_rd2            ml   0       NaN      NaN      NaN                   ⛔ INSUFFICIENT SAMPLE
                                 pseudo_home_pod_game     march_madness_rd1_rd2         total   0       NaN      NaN      NaN                   ⛔ INSUFFICIENT SAMPLE
                                 pseudo_road_opponent     march_madness_rd1_rd2           ats   0       NaN      NaN      NaN                   ⛔ INSUFFICIENT SAMPLE
                                 pseudo_road_opponent     march_madness_rd1_rd2            ml   0       NaN      NaN      NaN                   ⛔ INSUFFICIENT SAMPLE
                                 pseudo_road_opponent     march_madness_rd1_rd2         total   0       NaN      NaN      NaN                   ⛔ INSUFFICIENT SAMPLE
                                    both_true_neutral     march_madness_rd1_rd2           ats   0       NaN      NaN      NaN                   ⛔ INSUFFICIENT SAMPLE
                                    both_true_neutral     march_madness_rd1_rd2            ml   0       NaN      NaN      NaN                   ⛔ INSUFFICIENT SAMPLE
                                    both_true_neutral     march_madness_rd1_rd2         total   0       NaN      NaN      NaN                   ⛔ INSUFFICIENT SAMPLE
                                  travel_distance_gap     march_madness_rd1_rd2           ats   0       NaN      NaN      NaN                   ⛔ INSUFFICIENT SAMPLE
                                  travel_distance_gap     march_madness_rd1_rd2            ml   0       NaN      NaN      NaN                   ⛔ INSUFFICIENT SAMPLE
                                  travel_distance_gap     march_madness_rd1_rd2         total   0       NaN      NaN      NaN                   ⛔ INSUFFICIENT SAMPLE
                             first_meeting_cross_conf     march_madness_rd1_rd2           ats   0       NaN      NaN      NaN                   ⛔ INSUFFICIENT SAMPLE
                             first_meeting_cross_conf     march_madness_rd1_rd2            ml   0       NaN      NaN      NaN                   ⛔ INSUFFICIENT SAMPLE
                             first_meeting_cross_conf     march_madness_rd1_rd2         total   0       NaN      NaN      NaN                   ⛔ INSUFFICIENT SAMPLE
                                rd2_rematch_same_conf     march_madness_rd1_rd2           ats   0       NaN      NaN      NaN                   ⛔ INSUFFICIENT SAMPLE
                                rd2_rematch_same_conf     march_madness_rd1_rd2            ml   0       NaN      NaN      NaN                   ⛔ INSUFFICIENT SAMPLE
                                rd2_rematch_same_conf     march_madness_rd1_rd2         total   0       NaN      NaN      NaN                   ⛔ INSUFFICIENT SAMPLE
                         underdog_coach_upset_history     march_madness_rd1_rd2           ats   0       NaN      NaN      NaN                   ⛔ INSUFFICIENT SAMPLE
                         underdog_coach_upset_history     march_madness_rd1_rd2            ml   0       NaN      NaN      NaN                   ⛔ INSUFFICIENT SAMPLE
                         underdog_coach_upset_history     march_madness_rd1_rd2         total   0       NaN      NaN      NaN                   ⛔ INSUFFICIENT SAMPLE
                favorite_coach_low_tourney_experience     march_madness_rd1_rd2           ats   0       NaN      NaN      NaN                   ⛔ INSUFFICIENT SAMPLE
                favorite_coach_low_tourney_experience     march_madness_rd1_rd2            ml   0       NaN      NaN      NaN                   ⛔ INSUFFICIENT SAMPLE
                favorite_coach_low_tourney_experience     march_madness_rd1_rd2         total   0       NaN      NaN      NaN                   ⛔ INSUFFICIENT SAMPLE
                           efficiency_gap_under_8_rd1     march_madness_rd1_rd2           ats   0       NaN      NaN      NaN                   ⛔ INSUFFICIENT SAMPLE
                           efficiency_gap_under_8_rd1     march_madness_rd1_rd2            ml   0       NaN      NaN      NaN                   ⛔ INSUFFICIENT SAMPLE
                           efficiency_gap_under_8_rd1     march_madness_rd1_rd2         total   0       NaN      NaN      NaN                   ⛔ INSUFFICIENT SAMPLE
                           efficiency_gap_under_8_rd2     march_madness_rd1_rd2           ats   0       NaN      NaN      NaN                   ⛔ INSUFFICIENT SAMPLE
                           efficiency_gap_under_8_rd2     march_madness_rd1_rd2            ml   0       NaN      NaN      NaN                   ⛔ INSUFFICIENT SAMPLE
                           efficiency_gap_under_8_rd2     march_madness_rd1_rd2         total   0       NaN      NaN      NaN                   ⛔ INSUFFICIENT SAMPLE
                    underdog_physical_conf_vs_finesse     march_madness_rd1_rd2           ats   0       NaN      NaN      NaN                   ⛔ INSUFFICIENT SAMPLE
                    underdog_physical_conf_vs_finesse     march_madness_rd1_rd2            ml   0       NaN      NaN      NaN                   ⛔ INSUFFICIENT SAMPLE
                    underdog_physical_conf_vs_finesse     march_madness_rd1_rd2         total   0       NaN      NaN      NaN                   ⛔ INSUFFICIENT SAMPLE
                      underdog_elite_perimeter_scorer     march_madness_rd1_rd2           ats   0       NaN      NaN      NaN                   ⛔ INSUFFICIENT SAMPLE
                      underdog_elite_perimeter_scorer     march_madness_rd1_rd2            ml   0       NaN      NaN      NaN                   ⛔ INSUFFICIENT SAMPLE
                      underdog_elite_perimeter_scorer     march_madness_rd1_rd2         total   0       NaN      NaN      NaN                   ⛔ INSUFFICIENT SAMPLE
                underdog_tourney_experience_advantage     march_madness_rd1_rd2           ats   0       NaN      NaN      NaN                   ⛔ INSUFFICIENT SAMPLE
                underdog_tourney_experience_advantage     march_madness_rd1_rd2            ml   0       NaN      NaN      NaN                   ⛔ INSUFFICIENT SAMPLE
                underdog_tourney_experience_advantage     march_madness_rd1_rd2         total   0       NaN      NaN      NaN                   ⛔ INSUFFICIENT SAMPLE
                       public_75pct_plus_rd1_favorite     march_madness_rd1_rd2           ats   0       NaN      NaN      NaN                   ⛔ INSUFFICIENT SAMPLE
                       public_75pct_plus_rd1_favorite     march_madness_rd1_rd2            ml   0       NaN      NaN      NaN                   ⛔ INSUFFICIENT SAMPLE
                       public_75pct_plus_rd1_favorite     march_madness_rd1_rd2         total   0       NaN      NaN      NaN                   ⛔ INSUFFICIENT SAMPLE
                            reverse_line_movement_rd1     march_madness_rd1_rd2           ats   0       NaN      NaN      NaN                   ⛔ INSUFFICIENT SAMPLE
                            reverse_line_movement_rd1     march_madness_rd1_rd2            ml   0       NaN      NaN      NaN                   ⛔ INSUFFICIENT SAMPLE
                            reverse_line_movement_rd1     march_madness_rd1_rd2         total   0       NaN      NaN      NaN                   ⛔ INSUFFICIENT SAMPLE
                                 name_brand_inflation     march_madness_rd1_rd2           ats   0       NaN      NaN      NaN                   ⛔ INSUFFICIENT SAMPLE
                                 name_brand_inflation     march_madness_rd1_rd2            ml   0       NaN      NaN      NaN                   ⛔ INSUFFICIENT SAMPLE
                                 name_brand_inflation     march_madness_rd1_rd2         total   0       NaN      NaN      NaN                   ⛔ INSUFFICIENT SAMPLE
                    cinderella_narrative_overreaction     march_madness_rd1_rd2           ats   0       NaN      NaN      NaN                   ⛔ INSUFFICIENT SAMPLE
                    cinderella_narrative_overreaction     march_madness_rd1_rd2            ml   0       NaN      NaN      NaN                   ⛔ INSUFFICIENT SAMPLE
                    cinderella_narrative_overreaction     march_madness_rd1_rd2         total   0       NaN      NaN      NaN                   ⛔ INSUFFICIENT SAMPLE
                         rd1_both_fast_different_conf     march_madness_rd1_rd2           ats   0       NaN      NaN      NaN                   ⛔ INSUFFICIENT SAMPLE
                         rd1_both_fast_different_conf     march_madness_rd1_rd2            ml   0       NaN      NaN      NaN                   ⛔ INSUFFICIENT SAMPLE
                         rd1_both_fast_different_conf     march_madness_rd1_rd2         total   0       NaN      NaN      NaN                   ⛔ INSUFFICIENT SAMPLE
                       rd1_both_poor_defenses_neutral     march_madness_rd1_rd2           ats   0       NaN      NaN      NaN                   ⛔ INSUFFICIENT SAMPLE
                       rd1_both_poor_defenses_neutral     march_madness_rd1_rd2            ml   0       NaN      NaN      NaN                   ⛔ INSUFFICIENT SAMPLE
                       rd1_both_poor_defenses_neutral     march_madness_rd1_rd2         total   0       NaN      NaN      NaN                   ⛔ INSUFFICIENT SAMPLE
                          rd2_both_survived_close_rd1     march_madness_rd1_rd2           ats   0       NaN      NaN      NaN                   ⛔ INSUFFICIENT SAMPLE
                          rd2_both_survived_close_rd1     march_madness_rd1_rd2            ml   0       NaN      NaN      NaN                   ⛔ INSUFFICIENT SAMPLE
                          rd2_both_survived_close_rd1     march_madness_rd1_rd2         total   0       NaN      NaN      NaN                   ⛔ INSUFFICIENT SAMPLE
                     rd1_two_defensive_identity_teams     march_madness_rd1_rd2           ats   0       NaN      NaN      NaN                   ⛔ INSUFFICIENT SAMPLE
                     rd1_two_defensive_identity_teams     march_madness_rd1_rd2            ml   0       NaN      NaN      NaN                   ⛔ INSUFFICIENT SAMPLE
                     rd1_two_defensive_identity_teams     march_madness_rd1_rd2         total   0       NaN      NaN      NaN                   ⛔ INSUFFICIENT SAMPLE
                 rd1_conf_tourney_champ_fatigue_under     march_madness_rd1_rd2           ats   0       NaN      NaN      NaN                   ⛔ INSUFFICIENT SAMPLE
                 rd1_conf_tourney_champ_fatigue_under     march_madness_rd1_rd2            ml   0       NaN      NaN      NaN                   ⛔ INSUFFICIENT SAMPLE
                 rd1_conf_tourney_champ_fatigue_under     march_madness_rd1_rd2         total   0       NaN      NaN      NaN                   ⛔ INSUFFICIENT SAMPLE
                        underdog_mti_up_favorite_flat     march_madness_rd1_rd2           ats   0       NaN      NaN      NaN                   ⛔ INSUFFICIENT SAMPLE
                        underdog_mti_up_favorite_flat     march_madness_rd1_rd2            ml   0       NaN      NaN      NaN                   ⛔ INSUFFICIENT SAMPLE
                        underdog_mti_up_favorite_flat     march_madness_rd1_rd2         total   0       NaN      NaN      NaN                   ⛔ INSUFFICIENT SAMPLE
                   underdog_odi_defensive_elimination     march_madness_rd1_rd2           ats   0       NaN      NaN      NaN                   ⛔ INSUFFICIENT SAMPLE
                   underdog_odi_defensive_elimination     march_madness_rd1_rd2            ml   0       NaN      NaN      NaN                   ⛔ INSUFFICIENT SAMPLE
                   underdog_odi_defensive_elimination     march_madness_rd1_rd2         total   0       NaN      NaN      NaN                   ⛔ INSUFFICIENT SAMPLE
                          mid_major_sci_battle_tested     march_madness_rd1_rd2           ats   0       NaN      NaN      NaN                   ⛔ INSUFFICIENT SAMPLE
                          mid_major_sci_battle_tested     march_madness_rd1_rd2            ml   0       NaN      NaN      NaN                   ⛔ INSUFFICIENT SAMPLE
                          mid_major_sci_battle_tested     march_madness_rd1_rd2         total   0       NaN      NaN      NaN                   ⛔ INSUFFICIENT SAMPLE
                   low_spr_underdog_high_spr_favorite     march_madness_rd1_rd2           ats   0       NaN      NaN      NaN                   ⛔ INSUFFICIENT SAMPLE
                   low_spr_underdog_high_spr_favorite     march_madness_rd1_rd2            ml   0       NaN      NaN      NaN                   ⛔ INSUFFICIENT SAMPLE
                   low_spr_underdog_high_spr_favorite     march_madness_rd1_rd2         total   0       NaN      NaN      NaN                   ⛔ INSUFFICIENT SAMPLE
                      possession_dominance_tournament     march_madness_rd1_rd2           ats   0       NaN      NaN      NaN                   ⛔ INSUFFICIENT SAMPLE
                      possession_dominance_tournament     march_madness_rd1_rd2            ml   0       NaN      NaN      NaN                   ⛔ INSUFFICIENT SAMPLE
                      possession_dominance_tournament     march_madness_rd1_rd2         total   0       NaN      NaN      NaN                   ⛔ INSUFFICIENT SAMPLE
                          fast_efficient_vs_slow_poor     march_madness_rd1_rd2           ats   0       NaN      NaN      NaN                   ⛔ INSUFFICIENT SAMPLE
                          fast_efficient_vs_slow_poor     march_madness_rd1_rd2            ml   0       NaN      NaN      NaN                   ⛔ INSUFFICIENT SAMPLE
                          fast_efficient_vs_slow_poor     march_madness_rd1_rd2         total   0       NaN      NaN      NaN                   ⛔ INSUFFICIENT SAMPLE
                       low_tov_high_oreb_neutral_site     march_madness_rd1_rd2           ats   0       NaN      NaN      NaN                   ⛔ INSUFFICIENT SAMPLE
                       low_tov_high_oreb_neutral_site     march_madness_rd1_rd2            ml   0       NaN      NaN      NaN                   ⛔ INSUFFICIENT SAMPLE
                       low_tov_high_oreb_neutral_site     march_madness_rd1_rd2         total   0       NaN      NaN      NaN                   ⛔ INSUFFICIENT SAMPLE
                             seed_efficiency_mismatch     march_madness_rd1_rd2           ats   0       NaN      NaN      NaN                   ⛔ INSUFFICIENT SAMPLE
                             seed_efficiency_mismatch     march_madness_rd1_rd2            ml   0       NaN      NaN      NaN                   ⛔ INSUFFICIENT SAMPLE
                             seed_efficiency_mismatch     march_madness_rd1_rd2         total   0       NaN      NaN      NaN                   ⛔ INSUFFICIENT SAMPLE
                                  pes_full_pes_agrees            pes_validation       covered   0       NaN      NaN      NaN                   ⛔ INSUFFICIENT SAMPLE
                               pes_full_pes_disagrees            pes_validation       covered   0       NaN      NaN      NaN                   ⛔ INSUFFICIENT SAMPLE
                                  pes_full_pes_agrees            pes_validation        ml_won   0       NaN      NaN      NaN                   ⛔ INSUFFICIENT SAMPLE
                               pes_full_pes_disagrees            pes_validation        ml_won   0       NaN      NaN      NaN                   ⛔ INSUFFICIENT SAMPLE
                                  pes_full_pes_agrees            pes_validation total_covered   0       NaN      NaN      NaN                   ⛔ INSUFFICIENT SAMPLE
                               pes_full_pes_disagrees            pes_validation total_covered   0       NaN      NaN      NaN                   ⛔ INSUFFICIENT SAMPLE
                            pes_tournament_pes_agrees            pes_validation       covered   0       NaN      NaN      NaN                   ⛔ INSUFFICIENT SAMPLE
                         pes_tournament_pes_disagrees            pes_validation       covered   0       NaN      NaN      NaN                   ⛔ INSUFFICIENT SAMPLE
                            pes_tournament_pes_agrees            pes_validation        ml_won   0       NaN      NaN      NaN                   ⛔ INSUFFICIENT SAMPLE
                         pes_tournament_pes_disagrees            pes_validation        ml_won   0       NaN      NaN      NaN                   ⛔ INSUFFICIENT SAMPLE
                            pes_tournament_pes_agrees            pes_validation total_covered   0       NaN      NaN      NaN                   ⛔ INSUFFICIENT SAMPLE
                         pes_tournament_pes_disagrees            pes_validation total_covered   0       NaN      NaN      NaN                   ⛔ INSUFFICIENT SAMPLE
                          pes_enhancement_cutoff_None           pes_enhancement       covered 152  0.500000      NaN 0.500000                              ❌ NEGATIVE
                          pes_enhancement_cutoff_None           pes_enhancement       covered   0       NaN      NaN      NaN                   ⛔ INSUFFICIENT SAMPLE
                          pes_enhancement_cutoff_None           pes_enhancement       covered   0       NaN      NaN      NaN                   ⛔ INSUFFICIENT SAMPLE
                          pes_enhancement_cutoff_None           pes_enhancement       covered   0       NaN      NaN      NaN                   ⛔ INSUFFICIENT SAMPLE
                          pes_enhancement_cutoff_None           pes_enhancement       covered   0       NaN      NaN      NaN                   ⛔ INSUFFICIENT SAMPLE
                          pes_enhancement_cutoff_None           pes_enhancement       covered   0       NaN      NaN      NaN                   ⛔ INSUFFICIENT SAMPLE
                          pes_enhancement_cutoff_None           pes_enhancement       covered   0       NaN      NaN      NaN                   ⛔ INSUFFICIENT SAMPLE
                          pes_enhancement_cutoff_None           pes_enhancement       covered   0       NaN      NaN      NaN                   ⛔ INSUFFICIENT SAMPLE
                          pes_enhancement_cutoff_None           pes_enhancement       covered   0       NaN      NaN      NaN                   ⛔ INSUFFICIENT SAMPLE
                          pes_enhancement_cutoff_None           pes_enhancement       covered   0       NaN      NaN      NaN                   ⛔ INSUFFICIENT SAMPLE
                          pes_enhancement_cutoff_None           pes_enhancement        ml_won 152  0.500000      NaN 0.500000                              ❌ NEGATIVE
                          pes_enhancement_cutoff_None           pes_enhancement        ml_won   0       NaN      NaN      NaN                   ⛔ INSUFFICIENT SAMPLE
                          pes_enhancement_cutoff_None           pes_enhancement        ml_won   0       NaN      NaN      NaN                   ⛔ INSUFFICIENT SAMPLE
                          pes_enhancement_cutoff_None           pes_enhancement        ml_won   0       NaN      NaN      NaN                   ⛔ INSUFFICIENT SAMPLE
                          pes_enhancement_cutoff_None           pes_enhancement        ml_won   0       NaN      NaN      NaN                   ⛔ INSUFFICIENT SAMPLE
                          pes_enhancement_cutoff_None           pes_enhancement        ml_won   0       NaN      NaN      NaN                   ⛔ INSUFFICIENT SAMPLE
                          pes_enhancement_cutoff_None           pes_enhancement        ml_won   0       NaN      NaN      NaN                   ⛔ INSUFFICIENT SAMPLE
                          pes_enhancement_cutoff_None           pes_enhancement        ml_won   0       NaN      NaN      NaN                   ⛔ INSUFFICIENT SAMPLE
                          pes_enhancement_cutoff_None           pes_enhancement        ml_won   0       NaN      NaN      NaN                   ⛔ INSUFFICIENT SAMPLE
                          pes_enhancement_cutoff_None           pes_enhancement        ml_won   0       NaN      NaN      NaN                   ⛔ INSUFFICIENT SAMPLE
                          pes_enhancement_cutoff_None           pes_enhancement total_covered 152  0.539474      NaN 0.500000                              ❌ NEGATIVE
                          pes_enhancement_cutoff_None           pes_enhancement total_covered   0       NaN      NaN      NaN                   ⛔ INSUFFICIENT SAMPLE
                          pes_enhancement_cutoff_None           pes_enhancement total_covered   0       NaN      NaN      NaN                   ⛔ INSUFFICIENT SAMPLE
                          pes_enhancement_cutoff_None           pes_enhancement total_covered   0       NaN      NaN      NaN                   ⛔ INSUFFICIENT SAMPLE
                          pes_enhancement_cutoff_None           pes_enhancement total_covered   0       NaN      NaN      NaN                   ⛔ INSUFFICIENT SAMPLE
                          pes_enhancement_cutoff_None           pes_enhancement total_covered   0       NaN      NaN      NaN                   ⛔ INSUFFICIENT SAMPLE
                          pes_enhancement_cutoff_None           pes_enhancement total_covered   0       NaN      NaN      NaN                   ⛔ INSUFFICIENT SAMPLE
                          pes_enhancement_cutoff_None           pes_enhancement total_covered   0       NaN      NaN      NaN                   ⛔ INSUFFICIENT SAMPLE
                          pes_enhancement_cutoff_None           pes_enhancement total_covered   0       NaN      NaN      NaN                   ⛔ INSUFFICIENT SAMPLE
                          pes_enhancement_cutoff_None           pes_enhancement total_covered   0       NaN      NaN      NaN                   ⛔ INSUFFICIENT SAMPLE
                                 pes_flip_cutoff_None                  pes_flip       covered   0       NaN      NaN      NaN ⛔ INSUFFICIENT SAMPLE | VERY LOW SAMPLE
                                 pes_flip_cutoff_None                  pes_flip        ml_won   0       NaN      NaN      NaN ⛔ INSUFFICIENT SAMPLE | VERY LOW SAMPLE
                                 pes_flip_cutoff_None                  pes_flip total_covered   0       NaN      NaN      NaN ⛔ INSUFFICIENT SAMPLE | VERY LOW SAMPLE
                                  eight_nine_seed_dog          upset_validation            ml  20  0.550000 0.265481 0.011332               baseline_upset_rate=0.285
                             twelve_over_five_profile          upset_validation            ml  20  0.350000 0.065481 0.333526               baseline_upset_rate=0.285
```

## Failed

```text
                layer_name             scenario        market   n  hit_rate      lift  p_value                     notes
          elimination_game situational_underdog            ml  76  0.263158 -0.031197 0.763177                ❌ NEGATIVE
        mid_major_underdog situational_underdog           ats 248  0.475806  0.000000      NaN              ⚠️ REDUNDANT
        mid_major_underdog situational_underdog            ml 248  0.294355  0.000000      NaN              ⚠️ REDUNDANT
             pes_full_base       pes_validation       covered 496  0.500000  0.000000 0.500000                ❌ NEGATIVE
      pes_full_pes_neutral       pes_validation       covered 496  0.500000  0.000000 0.500000                ❌ NEGATIVE
             pes_full_base       pes_validation        ml_won 496  0.500000  0.000000 0.500000                ❌ NEGATIVE
      pes_full_pes_neutral       pes_validation        ml_won 496  0.500000  0.000000 0.500000                ❌ NEGATIVE
             pes_full_base       pes_validation total_covered 496  0.475806  0.000000 0.500000                ❌ NEGATIVE
      pes_full_pes_neutral       pes_validation total_covered 496  0.475806  0.000000 0.500000                ❌ NEGATIVE
       pes_tournament_base       pes_validation       covered 152  0.500000  0.000000 0.500000                ❌ NEGATIVE
pes_tournament_pes_neutral       pes_validation       covered 152  0.500000  0.000000 0.500000                ❌ NEGATIVE
       pes_tournament_base       pes_validation        ml_won 152  0.500000  0.000000 0.500000                ❌ NEGATIVE
pes_tournament_pes_neutral       pes_validation        ml_won 152  0.500000  0.000000 0.500000                ❌ NEGATIVE
       pes_tournament_base       pes_validation total_covered 152  0.539474  0.000000 0.500000                ❌ NEGATIVE
pes_tournament_pes_neutral       pes_validation total_covered 152  0.539474  0.000000 0.500000                ❌ NEGATIVE
     double_digit_seed_dog     upset_validation            ml 171  0.251462 -0.033057 0.851803 baseline_upset_rate=0.285
```

## Blocked / Theoretical Only

None.

## Recommended Actions
- VALIDATED: 1 layer(s) -> promote
- INCONCLUSIVE: 4 layer(s) -> monitor
- PROVISIONAL: 314 layer(s) -> revisit
- FAILED: 16 layer(s) -> retire
- BLOCKED: 0 layer(s) -> revisit after prerequisites
