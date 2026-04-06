# New_GlucoVector v8 Comprehensive Report

Data: `New_data/P1_final_with_D4_DI/P1_final`

## Config Comparison (D4 calibrated metrics)

| config                 |   sspg_n |   sspg_pearson_r |   sspg_r2 |   sspg_rmse |   sspg_mae |   di_n |   di_pearson_r |       di_r2 |   di_rmse |   di_mae |
|:-----------------------|---------:|-----------------:|----------:|------------:|-----------:|-------:|---------------:|------------:|----------:|---------:|
| Config_A_10D           |       20 |         0.653311 |  0.29845  |     66.8565 |    47.3398 |     16 |       0.564234 | -0.125252   |  0.769065 | 0.690951 |
| Config_B_16D_Recon     |       20 |         0.589123 |  0.251175 |     69.0723 |    48.5842 |     16 |       0.583841 |  0.00439065 |  0.723406 | 0.633566 |
| Config_C_16D_ODEHybrid |       20 |         0.749687 |  0.334796 |     65.1016 |    45.8487 |     16 |       0.654843 | -0.107277   |  0.762897 | 0.695764 |
| Config_D_26D_Original  |       20 |         0.821999 |  0.118406 |     74.946  |    50.6691 |     16 |       0.671494 | -0.222805   |  0.801709 | 0.731788 |

## SHAP Top-5 Features (SSPG)

- **Config_A_10D**: z_init_0 (83.0%); Gb (13.0%); mi (2.1%); z_init_2 (1.0%); tau_m (0.5%)
- **Config_B_16D_Recon**: z_nonseq_2 (15.4%); z_nonseq_12 (13.4%); z_nonseq_3 (12.9%); z_nonseq_1 (12.1%); z_nonseq_11 (11.6%)
- **Config_C_16D_ODEHybrid**: z_init_0 (61.2%); z_nonseq_3 (7.8%); z_nonseq_6 (7.2%); z_nonseq_2 (6.0%); z_nonseq_13 (3.6%)
- **Config_D_26D_Original**: z_init_0 (32.5%); Gb (31.9%); z_init_3 (10.2%); tau_m (7.5%); z_nonseq_12 (2.4%)

## D4 Meal-Type Comparison (best SSPG Pearson per config)

| config                 | best_meal_type   |   sspg_pearson_r |   sspg_rmse |
|:-----------------------|:-----------------|-----------------:|------------:|
| Config_A_10D           | Protein_bar      |         0.786851 |     65.3436 |
| Config_B_16D_Recon     | Protein_bar      |         0.684784 |     68.0652 |
| Config_C_16D_ODEHybrid | Protein_bar      |         0.783931 |     68.1744 |
| Config_D_26D_Original  | Protein_bar      |         0.843563 |     72.2844 |