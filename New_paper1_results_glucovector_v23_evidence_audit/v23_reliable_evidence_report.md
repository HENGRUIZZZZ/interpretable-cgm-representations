# v23 Reliable Evidence Report

Generated: 2026-04-08T01:11:13.641725

## 1) All Usable Data (D1-D4)

| dataset     | file                | exists   |   n_rows |   n_cols | columns                                                                                                                                                                                                                                                                         |   n_subjects |   n_meal_types | time_min            | time_max            |
|:------------|:--------------------|:---------|---------:|---------:|:--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-------------:|---------------:|:--------------------|:--------------------|
| D1_metwally | subjects.csv        | True     |       59 |       14 | subject_id,original_id,age,bmi,sex,ethnicity,hba1c,fpg,ogtt_2h,cohort,has_cgm,has_gold_standard_public,dataset,weight_kg                                                                                                                                                        |           59 |            nan | nan                 | nan                 |
| D1_metwally | meals.csv           | True     |      148 |        6 | subject_id,meal_type,meal_context,carb_g,description,dataset                                                                                                                                                                                                                    |           59 |              1 | nan                 | nan                 |
| D1_metwally | cgm.csv             | True     |     4479 |        6 | subject_id,timepoint_mins,glucose_mg_dl,source,data_type,dataset                                                                                                                                                                                                                |           59 |            nan | nan                 | nan                 |
| D1_metwally | labels.csv          | True     |       59 |       25 | subject_id,dataset,demo_ExperimentType,demo_Age,demo_BMI,demo_Sex,demo_Ethnicity,demo_OGTT_2h,demo_HbA1c,demo_FPG,SSPG,MuscleIR_Class,DI,BetaCellFunction_Class,IE,IE_Class,Hepatic_IR_Index,HepaticIR_Class,GLP1_120min,GIP_120min,HOMA_B,HOMA_IR,HOMA_S,Matsuda_Index,T2D_PRS |           59 |            nan | nan                 | nan                 |
| D2_stanford | subjects.csv        | True     |       74 |       16 | subject_id,age,sex,ethnicity,bmi,hba1c,fasting_glucose,fasting_insulin,systolic_bp,diastolic_bp,total_cholesterol,ldl,hdl,free_fatty_acids,ogtt120,dataset                                                                                                                      |           74 |            nan | nan                 | nan                 |
| D2_stanford | meals.csv           | True     |      332 |        9 | subject_id,meal_type,n_readings,description,meal_category,carb_g,fat_g,protein_g,fiber_g                                                                                                                                                                                        |           38 |             24 | nan                 | nan                 |
| D2_stanford | cgm.csv             | True     |    23520 |       10 | glucose_mg_dl,subject_id,meal_type,mitigator,food_detail,rep,mins_since_start,minutes_after_meal,source,data_type                                                                                                                                                               |           38 |             24 | nan                 | nan                 |
| D2_stanford | labels.csv          | True     |       74 |       12 | subject_id,SSPG,DI,IE,Hepatic_IR,HOMA_IR,HOMA_B,hba1c,fasting_glucose,fasting_insulin,ogtt120,dataset                                                                                                                                                                           |           74 |            nan | nan                 | nan                 |
| D3_cgmacros | subjects.csv        | True     |       45 |        8 | subject_id,age,sex,bmi,weight_kg,height_cm,ethnicity,dataset                                                                                                                                                                                                                    |           45 |            nan | nan                 | nan                 |
| D3_cgmacros | meals.csv           | True     |     1706 |        9 | subject_id,timestamp,meal_type,calories_kcal,carb_g,protein_g,fat_g,fiber_g,amount_consumed_pct                                                                                                                                                                                 |           45 |             10 | 2019-11-16 10:03:00 | 2025-11-08 16:23:00 |
| D3_cgmacros | cgm.csv             | True     |  1317185 |        6 | subject_id,timestamp,glucose_mg_dl,sensor,source,data_type                                                                                                                                                                                                                      |           45 |            nan | 2019-11-15 20:12:00 | 2025-11-10 06:18:00 |
| D3_cgmacros | labels.csv          | True     |       45 |       17 | subject_id,hba1c,fasting_glucose_mg_dl,fasting_insulin_uiu_ml,triglycerides_mg_dl,total_cholesterol_mg_dl,hdl_mg_dl,non_hdl_mg_dl,ldl_mg_dl,vldl_mg_dl,cho_hdl_ratio,fingerstick_glucose_1,fingerstick_glucose_2,fingerstick_glucose_3,HOMA_IR,HOMA_B,dataset                   |           45 |            nan | nan                 | nan                 |
| D4_hall     | subjects.csv        | True     |       57 |       12 | subject_id,dataset_id,original_id,age,bmi,height_cm,weight_kg,hba1c,diagnosis,glucotype,sex,ethnicity                                                                                                                                                                           |           57 |            nan | nan                 | nan                 |
| D4_hall     | meals.csv           | True     |      240 |       11 | subject_id,timestamp,meal_type,meal_code,description,repeat,meal_category,carb_g,fat_g,protein_g,fiber_g                                                                                                                                                                        |           66 |              4 | 2013-11-11 00:00:00 | 2017-08-10 00:00:00 |
| D4_hall     | cgm.csv             | True     |     6152 |        6 | meal_context,subject_id,timestamp,glucose_mg_dl,source,data_type                                                                                                                                                                                                                |           30 |            nan | 2016-08-04 08:10:00 | 2017-07-20 08:00:00 |
| D4_hall     | labels.csv          | True     |       57 |       19 | subject_id,hba1c,fasting_glucose_mg_dl,fasting_insulin,ogtt_2h_mgdl,sspg,ie,homa_ir,homa_b,hs_crp,total_cholesterol,triglycerides,hdl,ldl,insulin_rate_dd,diagnosis,glucotype,oral_di,DI                                                                                        |           57 |            nan | nan                 | nan                 |
| D4_hall     | oral_di.csv         | True     |       49 |        5 | d4_id,d3h_id,matsuda,igi,oral_di                                                                                                                                                                                                                                                |          nan |            nan | nan                 | nan                 |
| D4_hall     | ogtt_timeseries.csv | True     |     2368 |        5 | subject_id,date,timepoint_mins,value,parameter                                                                                                                                                                                                                                  |           74 |            nan | nan                 | nan                 |
| D4_hall     | cgm_freeliving.csv  | True     |   105426 |        3 | subject_id,timestamp,glucose_mgdl                                                                                                                                                                                                                                               |           57 |            nan | 2014-02-03 03:42:12 | 2017-07-11 20:41:32 |

## 2) Subject Overlap Audit

| dataset_a   | dataset_b   |   n_overlap_subjects |
|:------------|:------------|---------------------:|
| D1_metwally | D1_metwally |                   59 |
| D1_metwally | D2_stanford |                    0 |
| D1_metwally | D3_cgmacros |                    0 |
| D1_metwally | D4_hall     |                    0 |
| D2_stanford | D1_metwally |                    0 |
| D2_stanford | D2_stanford |                   74 |
| D2_stanford | D3_cgmacros |                    0 |
| D2_stanford | D4_hall     |                    0 |
| D3_cgmacros | D1_metwally |                    0 |
| D3_cgmacros | D2_stanford |                    0 |
| D3_cgmacros | D3_cgmacros |                   45 |
| D3_cgmacros | D4_hall     |                    0 |
| D4_hall     | D1_metwally |                    0 |
| D4_hall     | D2_stanford |                    0 |
| D4_hall     | D3_cgmacros |                    0 |
| D4_hall     | D4_hall     |                   57 |

## 3) Experiment Registry (completed)

| version   | experiment                        | data_scope       | type                    | status    |
|:----------|:----------------------------------|:-----------------|:------------------------|:----------|
| v18       | Exp1_Wang_Baseline                | D1+D2->D4        | regression              | completed |
| v18       | Exp2_Metwally14                   | D1+D2->D4        | regression              | completed |
| v18       | Exp3_Healey                       | D1+D2->D4        | regression              | completed |
| v18       | Exp4_SimpleStats                  | D1+D2->D4        | regression              | completed |
| v18       | Exp8_GV_CorrLoss                  | D1+D2->D4        | regression              | completed |
| v19       | Joint_Classification              | D1+D2->D4        | classification          | completed |
| v20       | Fairness_Healey_CGMOnly           | D1+D2->D4        | fairness                | completed |
| v20       | D3_FreeLiving_Benchmark           | D3               | free-living             | completed |
| v21       | Ridge10D_26D_Optimization         | D1+D2->D4        | retrain_head            | completed |
| v21       | Stack_Diagnostics                 | D1+D2->D4        | failure_analysis        | completed |
| v22       | Locked_Protocol_Primary_Secondary | D1+D2->D4        | locked_protocol         | completed |
| v22       | 16D_Conditional_Utility           | D4_meal_level    | conditional_utility     | completed |
| v22       | Beyond_Metwally_Residual          | D4_subject_level | residual_explainability | completed |

## 4) Claim-Evidence Matrix

| claim                                            | source                               | metric                    | value                 | status        |
|:-------------------------------------------------|:-------------------------------------|:--------------------------|:----------------------|:--------------|
| 26D improves SSPG ranking vs 10D                 | v22_paired_bootstrap_deltas.csv      | delta_spearman            | 0.170 [-0.023,0.506]  | trend_only    |
| 26D beats Metwally on SSPG ranking               | v22_paired_bootstrap_deltas.csv      | delta_spearman            | -0.048 [-0.272,0.150] | not_confirmed |
| 26D improves IR AUROC vs Metwally                | v22_secondary_clinical_endpoints.csv | delta_ir_auc              | 0.017                 | supported     |
| 26D improves Decomp AUROC vs Metwally            | v22_secondary_clinical_endpoints.csv | delta_decomp_auc          | 0.064                 | supported     |
| 16D utility increases in high-uncertainty meals  | v22_16d_conditional_utility.csv      | win26_rate_high_minus_low | 0.211                 | supported     |
| 16D predicts residuals not explained by Metwally | v22_beyond_metwally_summary.json     | residual_spearman         | 0.517                 | supported     |

## 5) Completeness Checklist

| item                          | exists   | path                                                                                                                                                        |
|:------------------------------|:---------|:------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Data inventory D1-D4          | True     | /Users/hertz1030/Documents/GitHub/interpretable-cgm-representations/New_paper1_results_glucovector_v23_evidence_audit/v23_data_inventory_d1_to_d4.csv       |
| Subject overlap audit         | True     | /Users/hertz1030/Documents/GitHub/interpretable-cgm-representations/New_paper1_results_glucovector_v23_evidence_audit/v23_subject_overlap_matrix.csv        |
| Locked primary endpoints      | True     | /Users/hertz1030/Documents/GitHub/interpretable-cgm-representations/New_paper1_results_glucovector_v22_locked_protocol/v22_primary_endpoints_locked.csv     |
| Locked secondary endpoints    | True     | /Users/hertz1030/Documents/GitHub/interpretable-cgm-representations/New_paper1_results_glucovector_v22_locked_protocol/v22_secondary_clinical_endpoints.csv |
| Paired bootstrap deltas       | True     | /Users/hertz1030/Documents/GitHub/interpretable-cgm-representations/New_paper1_results_glucovector_v22_locked_protocol/v22_paired_bootstrap_deltas.csv      |
| 16D conditional utility       | True     | /Users/hertz1030/Documents/GitHub/interpretable-cgm-representations/New_paper1_results_glucovector_v22_locked_protocol/v22_16d_conditional_utility.csv      |
| Beyond-Metwally residual test | True     | /Users/hertz1030/Documents/GitHub/interpretable-cgm-representations/New_paper1_results_glucovector_v22_locked_protocol/v22_beyond_metwally_summary.json     |
