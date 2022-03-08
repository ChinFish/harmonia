import pandas as pd
import os
import numpy as np
import warnings
warnings.filterwarnings("ignore")
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import OneHotEncoder
from imblearn.under_sampling import RandomUnderSampler

def ischemic_stroke(df):
    df_1 = df[df["icd_id"].isin([1, 2])]
    print(df_1.shape)
    return df_1

def feature_selection(df):
    df_1 = df[selected_feature]
    print(df_1.shape)
    return df_1

def categorical_features(df, nom_f, ord_f, bl_f, b_i, ni_in, ni_out):
    # nominal_features
    df["gender_tx"][df["gender_tx"] == "M"] = 1
    df["gender_tx"][df["gender_tx"] == "F"] = 0

    for i in df[nom_f]:
        df[i] = pd.to_numeric(df[i], errors="coerce")
    df["opc_id"][~df["opc_id"].isin([1, 2, 3])] = np.nan
    df["toast_id"][~df["toast_id"].isin([1, 2, 3, 4, 5])] = np.nan
    df["offdt_id"][~df["offdt_id"].isin([1, 2, 3, 4, 5])] = np.nan
    df["gender_tx"][~df["gender_tx"].isin([0, 1])] = np.nan

    for i in df.loc[:, "hd_id":"ca_id"]:
        df[i] = pd.to_numeric(df[i], errors="coerce")
        df[i][~df[i].isin([0, 1, 2])] = np.nan

    for i in df.loc[:, "fahiid_parents_1":"fahiid_parents_4"]:
        df[i] = pd.to_numeric(df[i], errors="coerce")
        df[i][~df[i].isin([0, 1, 2])] = np.nan

    for i in df.loc[:, "fahiid_brsi_1":"fahiid_brsi_4"]:
        df[i] = pd.to_numeric(df[i], errors="coerce")
        df[i][~df[i].isin([0, 1, 2, 9])] = np.nan

    # ordinal_features
    for i in df[ord_f]:
        df[i] = pd.to_numeric(df[i], errors="coerce")

    df["discharged_mrs"][~df["discharged_mrs"].isin([0, 1, 2, 3, 4, 5, 6])] = np.nan
    df["gcse_nm"][~df["gcse_nm"].isin([1, 2, 3, 4])] = np.nan
    df["gcsv_nm"][~df["gcsv_nm"].isin([1, 2, 3, 4, 5])] = np.nan
    df["gcsm_nm"][~df["gcsm_nm"].isin([1, 2, 3, 4, 5, 6])] = np.nan

    # boolean
    for i in df[bl_f]:
        df[i].replace("1", 1, inplace=True)
        df[i].replace("0", 0, inplace=True)
        df[i].replace("Y", 1, inplace=True)
        df[i].replace("N", 0, inplace=True)
        df[i][~df[i].isin([0, 1])] = np.nan

    # barthel
    for i in df[b_i]:
        df[i] = pd.to_numeric(df[i], errors="coerce")

    df["feeding"][~df["feeding"].isin([0, 5, 10])] = np.nan
    df["transfers"][~df["transfers"].isin([0, 5, 10, 15])] = np.nan
    df["bathing"][~df["bathing"].isin([0, 5])] = np.nan
    df["toilet_use"][~df["toilet_use"].isin([0, 5, 10])] = np.nan
    df["grooming"][~df["grooming"].isin([0, 5])] = np.nan
    df["mobility"][~df["mobility"].isin([0, 5, 10, 15])] = np.nan
    df["stairs"][~df["stairs"].isin([0, 5, 10])] = np.nan
    df["dressing"][~df["dressing"].isin([0, 5, 10])] = np.nan
    df["bowel_control"][~df["bowel_control"].isin([0, 5, 10])] = np.nan
    df["bladder_control"][~df["bladder_control"].isin([0, 5, 10])] = np.nan

    # nihss_in
    for i in df[ni_in]:
        df[i] = pd.to_numeric(df[i], errors="coerce")

    df["nihs_1a_in"][(df["nihs_1a_in"] < 0) | (df["nihs_1a_in"] > 3)] = np.nan
    df["nihs_1b_in"][(df["nihs_1b_in"] < 0) | (df["nihs_1b_in"] > 2)] = np.nan
    df["nihs_1c_in"][(df["nihs_1c_in"] < 0) | (df["nihs_1c_in"] > 2)] = np.nan
    df["nihs_2_in"][(df["nihs_2_in"] < 0) | (df["nihs_2_in"] > 2)] = np.nan
    df["nihs_3_in"][(df["nihs_3_in"] < 0) | (df["nihs_3_in"] > 3)] = np.nan
    df["nihs_4_in"][(df["nihs_4_in"] < 0) | (df["nihs_4_in"] > 3)] = np.nan
    df["nihs_5al_in"][(df["nihs_5al_in"] < 0) | (df["nihs_5al_in"] > 4)] = np.nan
    df["nihs_5br_in"][(df["nihs_5br_in"] < 0) | (df["nihs_5br_in"] > 4)] = np.nan
    df["nihs_6al_in"][(df["nihs_6al_in"] < 0) | (df["nihs_6al_in"] > 4)] = np.nan
    df["nihs_6br_in"][(df["nihs_6br_in"] < 0) | (df["nihs_6br_in"] > 4)] = np.nan
    df["nihs_7_in"][(df["nihs_7_in"] < 0) | (df["nihs_7_in"] > 2)] = np.nan
    df["nihs_8_in"][(df["nihs_8_in"] < 0) | (df["nihs_8_in"] > 2)] = np.nan
    df["nihs_9_in"][(df["nihs_9_in"] < 0) | (df["nihs_9_in"] > 3)] = np.nan
    df["nihs_10_in"][(df["nihs_10_in"] < 0) | (df["nihs_10_in"] > 2)] = np.nan
    df["nihs_11_in"][(df["nihs_11_in"] < 0) | (df["nihs_11_in"] > 2)] = np.nan

    # nihss_out
    for i in df[ni_out]:
        df[i] = pd.to_numeric(df[i], errors="coerce")

    df["nihs_1a_out"][(df["nihs_1a_out"] < 0) | (df["nihs_1a_out"] > 3)] = np.nan
    df["nihs_1b_out"][(df["nihs_1b_out"] < 0) | (df["nihs_1b_out"] > 2)] = np.nan
    df["nihs_1c_out"][(df["nihs_1c_out"] < 0) | (df["nihs_1c_out"] > 2)] = np.nan
    df["nihs_2_out"][(df["nihs_2_out"] < 0) | (df["nihs_2_out"] > 2)] = np.nan
    df["nihs_3_out"][(df["nihs_3_out"] < 0) | (df["nihs_3_out"] > 3)] = np.nan
    df["nihs_4_out"][(df["nihs_4_out"] < 0) | (df["nihs_4_out"] > 3)] = np.nan
    df["nihs_5al_out"][(df["nihs_5al_out"] < 0) | (df["nihs_5al_out"] > 4)] = np.nan
    df["nihs_5br_out"][(df["nihs_5br_out"] < 0) | (df["nihs_5br_out"] > 4)] = np.nan
    df["nihs_6al_out"][(df["nihs_6al_out"] < 0) | (df["nihs_6al_out"] > 4)] = np.nan
    df["nihs_6br_out"][(df["nihs_6br_out"] < 0) | (df["nihs_6br_out"] > 4)] = np.nan
    df["nihs_7_out"][(df["nihs_7_out"] < 0) | (df["nihs_7_out"] > 2)] = np.nan
    df["nihs_8_out"][(df["nihs_8_out"] < 0) | (df["nihs_8_out"] > 2)] = np.nan
    df["nihs_9_out"][(df["nihs_9_out"] < 0) | (df["nihs_9_out"] > 3)] = np.nan
    df["nihs_10_out"][(df["nihs_10_out"] < 0) | (df["nihs_10_out"] > 2)] = np.nan
    df["nihs_11_out"][(df["nihs_11_out"] < 0) | (df["nihs_11_out"] > 2)] = np.nan

    df_1 = df.dropna(how='any', subset=nom_f+ord_f+bl_f+b_i+ni_in+ni_out).reset_index(drop=True)
    print(df_1.shape)
    df_1[ord_f] = OrdinalEncoder().fit_transform(df_1[ord_f])
    df_1[b_i] = OrdinalEncoder().fit_transform(df_1[b_i])
    df_1[ni_in] = OrdinalEncoder().fit_transform(df_1[ni_in])
    df_1[ni_out] = OrdinalEncoder().fit_transform(df_1[ni_out])

    ohe = OneHotEncoder(sparse=False, handle_unknown="ignore")
    nominal_ohe = pd.DataFrame(ohe.fit_transform(df_1[nom_f]))
    nominal_ohe.columns = ohe.get_feature_names(nom_f)
    df_2 = pd.concat([df_1, nominal_ohe], axis=1)
    df_2 = df_2.drop(nom_f, axis=1)
    print(df_2.shape)
    return df_2

def continuous_features(df, cont):
    # continuous
    for i in df[cont]:
        df[i] = pd.to_numeric(df[i], errors="coerce")
        df[i][df[i] == 999.9] = np.nan
        q1 = df[i].quantile(0.25)
        q3 = df[i].quantile(0.75)
        iqr = q3 - q1
        inner_fence = 1.5 * iqr

        inner_fence_low = q1 - inner_fence
        inner_fence_upp = q3 + inner_fence
        df[i][(df[i] < inner_fence_low) | (df[i] > inner_fence_upp)] = np.nan
        df[i][df[i] < 0] = np.nan

    df[cont] = df[cont].fillna(df[cont].median())
    df[cont] = MinMaxScaler().fit_transform(df[cont])
    print(df.shape)
    return df

def mRS_cleaning(df):
    df["mrs_tx_1"] = pd.to_numeric(df["mrs_tx_1"], errors="coerce")
    df["mrs_tx_1"][~df["mrs_tx_1"].isin([0, 1, 2, 3, 4, 5, 6, 9])] = np.nan

    df["mrs_tx_3"] = pd.to_numeric(df["mrs_tx_3"], errors="coerce")
    df["mrs_tx_3"][~df["mrs_tx_3"].isin([0, 1, 2, 3, 4, 5, 6, 9])] = np.nan

    df_1 = df.dropna(how='any', subset=["mrs_tx_1", "mrs_tx_3"])
    df_1["mrs_tx_3"][df_1["mrs_tx_3"].isin([0,1,2])] = 0 ## GOOD
    df_1["mrs_tx_3"][df_1["mrs_tx_3"].isin([3,4,5,6,9])] = 1 ## POOR
    print(df_1.shape)
    return df_1

def sampling(df):
    df_x = df.drop(["mrs_tx_3"], axis=1)
    df_y = df["mrs_tx_3"]
    rus = RandomUnderSampler(sampling_strategy='majority')
    df_x_rus, df_y_rus = rus.fit_resample(df_x, df_y)
    df_1 = pd.concat([df_x_rus, df_y_rus], axis=1)
    print(df_1.shape)
    return df_1

def preprocessing(df):
    # extract ischemic stroke from the whole database
    df_1 = ischemic_stroke(df)
    # subjectively select the wanted features
    df_2 = feature_selection(df_1)
    # clean categorical features by amputating undefined values and executing OneHot and Ordinal encoding
    df_3 = categorical_features(df_2, nominal_features, ordinal_features, boolean, barthel, nihss_in,nihss_out)
    # clean continuous features by amputating outliers and executing MinMaxScaler
    df_4 = continuous_features(df_3, continuous)
    # clean the outcome by deleting observations without values of mRS_3 and grouping into GOOD and POOR
    df_5 = mRS_cleaning(df_4)
    # if needed, under-sampling data to tackle imbalanced issue
    #df_6 = sampling(df_5)
    return df_5

if __name__ == '__main__':
    selected_feature = ["icase_id", "idcase_id", "mrs_tx_1", "mrs_tx_3", "height_nm", "weight_nm", "opc_id", "gcse_nm",
                        "gcsv_nm", "gcsm_nm", "sbp_nm", "dbp_nm", "bt_nm", "hr_nm", "rr_nm", "toast_id", "toastle_fl",
                        "toastli_fl", "toastsce_fl", "toastsmo_fl", "toastsra_fl", "toastsdi_fl", "toastsmi_fl",
                        "toastsantip_fl", "toastsau_fl", "toastshy_fl", "toastspr_fl", "toastsantit_fl", "toastsho_fl",
                        "toastshys_fl", "toastsca_fl", "thda_fl", "thdh_fl", "thdi_fl", "thdam_fl", "thdv_fl",
                        "thde_fl", "thdm_fl", "thdr_fl", "thdp_fl", "hb_nm", "hct_nm", "platelet_nm", "wbc_nm",
                        "ptt1_nm", "ptt2_nm", "ptinr_nm", "er_nm", "bun_nm", "cre_nm", "ua_nm", "tcho_nm", "tg_nm",
                        "hdl_nm", "ldl_nm", "gpt_nm", "trman_fl", "trmas_fl", "trmti_fl", "trmhe_fl", "trmwa_fl",
                        "trmia_fl", "trmfo_fl", "trmta_fl", "trmsd_fl", "trmre_fl", "trmen_fl", "trmag_fl", "trmcl_fl",
                        "trmpl_fl", "trmlm_fl", "trmiv_fl", "trmve_fl", "trmng_fl", "trmdy_fl", "trmicu_fl", "trmsm_fl",
                        "trmed_fl", "trmop_fl", "om_fl", "omas_fl", "omag_fl", "omti_fl", "omcl_fl", "omwa_fl",
                        "ompl_fl", "omanh_fl", "omand_fl", "omli_fl", "am_fl", "amas_fl", "amag_fl", "amti_fl",
                        "amcl_fl", "amwa_fl", "ampl_fl", "amanh_fl", "amand_fl", "amli_fl", "compn_fl", "comut_fl",
                        "comug_fl", "compr_fl", "compu_fl", "comac_fl", "comse_fl", "comde_fl", "detst_fl", "dethe_fl",
                        "detho_fl", "detha_fl", "detva_fl", "detre_fl", "detme_fl", "offdt_id", "ct_fl", "mri_fl",
                        "ecgl_fl", "ecga_fl", "ecgq_fl", "feeding", "transfers", "bathing", "toilet_use", "grooming",
                        "mobility", "stairs", "dressing", "bowel_control", "bladder_control", "discharged_mrs",
                        "cortical_aca_ctr", "cortical_mca_ctr", "subcortical_aca_ctr", "subcortical_mca_ctr",
                        "pca_cortex_ctr", "thalamus_ctr", "brainstem_ctr", "cerebellum_ctr", "watershed_ctr",
                        "hemorrhagic_infarct_ctr", "old_stroke_ctci", "cortical_aca_ctl", "cortical_mca_ctl",
                        "subcortical_aca_ctl", "subcortical_mca_ctl", "pca_cortex_ctl", "thalamus_ctl", "brainstem_ctl",
                        "cerebellum_ctl", "watershed_ctl", "hemorrhagic_infarct_ctl", "old_stroke_ctch",
                        "cortical_aca_mrir", "cortical_mca_mrir", "subcortical_aca_mrir", "subcortical_mca_mrir",
                        "pca_cortex_mrir", "thalamus_mrir", "brainstem_mrir", "cerebellum_mrir", "watershed_mrir",
                        "hemorrhagic_infarct_mrir", "old_stroke_mrici", "cortical_aca_mril", "cortical_mca_mril",
                        "subcortical_aca_mril", "subcortical_mca_mril", "pca_cortex_mril", "thalamus_mril",
                        "brainstem_mril", "cerebellum_mril", "watershed_mril", "hemorrhagic_infarct_mril",
                        "old_stroke_mrich", "hd_id", "pcva_id", "pcvaci_id", "pcvach_id", "po_id", "ur_id", "sm_id",
                        "ptia_id", "hc_id", "hcht_id", "hchc_id", "ht_id", "dm_id", "pad_id", "al_id", "ca_id",
                        "fahiid_parents_1", "fahiid_parents_2", "fahiid_parents_3", "fahiid_parents_4", "fahiid_brsi_1",
                        "fahiid_brsi_2", "fahiid_brsi_3", "fahiid_brsi_4", "nihs_1a_in", "nihs_1b_in", "nihs_1c_in",
                        "nihs_2_in", "nihs_3_in", "nihs_4_in", "nihs_5al_in", "nihs_5br_in", "nihs_6al_in",
                        "nihs_6br_in", "nihs_7_in", "nihs_8_in", "nihs_9_in", "nihs_10_in", "nihs_11_in", "nihs_1a_out",
                        "nihs_1b_out", "nihs_1c_out", "nihs_2_out", "nihs_3_out", "nihs_4_out", "nihs_5al_out",
                        "nihs_5br_out", "nihs_6al_out", "nihs_6br_out", "nihs_7_out", "nihs_8_out", "nihs_9_out",
                        "nihs_10_out", "nihs_11_out", "gender_tx", "age", "hospitalised_time"]

    nominal_features = ["opc_id", "toast_id", "offdt_id", "gender_tx", "hd_id", "pcva_id",
                        "pcvaci_id", "pcvach_id", "po_id", "ur_id", "sm_id", "ptia_id", "hc_id", "hcht_id",
                        "hchc_id", "ht_id", "dm_id", "pad_id", "al_id", "ca_id", "fahiid_parents_1",
                        "fahiid_parents_2", "fahiid_parents_3", "fahiid_parents_4", "fahiid_brsi_1",
                        "fahiid_brsi_2", "fahiid_brsi_3", "fahiid_brsi_4"]

    ordinal_features = ["gcse_nm", "gcsv_nm", "gcsm_nm", "discharged_mrs"]

    boolean = ["toastle_fl", "toastli_fl", "toastsce_fl", "toastsmo_fl", "toastsra_fl", "toastsdi_fl",
               "toastsmi_fl", "toastsantip_fl", "toastsau_fl", "toastshy_fl", "toastspr_fl", "toastsantit_fl",
               "toastsho_fl", "toastshys_fl", "toastsca_fl", "thda_fl", "thdh_fl", "thdi_fl", "thdam_fl", "thdv_fl",
               "thde_fl", "thdm_fl", "thdr_fl", "thdp_fl", "trman_fl", "trmas_fl", "trmti_fl", "trmhe_fl",
               "trmwa_fl", "trmia_fl", "trmfo_fl", "trmta_fl", "trmsd_fl", "trmre_fl", "trmen_fl", "trmag_fl",
               "trmcl_fl", "trmpl_fl", "trmlm_fl", "trmiv_fl", "trmve_fl", "trmng_fl", "trmdy_fl", "trmicu_fl",
               "trmsm_fl", "trmed_fl", "trmop_fl", "om_fl", "omas_fl", "omag_fl", "omti_fl", "omcl_fl", "omwa_fl",
               "ompl_fl", "omanh_fl", "omand_fl", "omli_fl", "am_fl", "amas_fl", "amag_fl", "amti_fl", "amcl_fl",
               "amwa_fl", "ampl_fl", "amanh_fl", "amand_fl", "amli_fl", "compn_fl", "comut_fl", "comug_fl",
               "compr_fl", "compu_fl", "comac_fl", "comse_fl", "comde_fl", "detst_fl", "dethe_fl", "detho_fl",
               "detha_fl", "detva_fl", "detre_fl", "detme_fl", "ct_fl", "mri_fl", "ecgl_fl", "ecga_fl", "ecgq_fl",
               "cortical_aca_ctr", "cortical_mca_ctr", "subcortical_aca_ctr", "subcortical_mca_ctr", "pca_cortex_ctr",
               "thalamus_ctr", "brainstem_ctr", "cerebellum_ctr", "watershed_ctr", "hemorrhagic_infarct_ctr",
               "old_stroke_ctci", "cortical_aca_ctl", "cortical_mca_ctl", "subcortical_aca_ctl", "subcortical_mca_ctl",
               "pca_cortex_ctl", "thalamus_ctl", "brainstem_ctl", "cerebellum_ctl", "watershed_ctl",
               "hemorrhagic_infarct_ctl", "old_stroke_ctch", "cortical_aca_mrir", "cortical_mca_mrir",
               "subcortical_aca_mrir", "subcortical_mca_mrir", "pca_cortex_mrir", "thalamus_mrir", "brainstem_mrir",
               "cerebellum_mrir", "watershed_mrir", "hemorrhagic_infarct_mrir", "old_stroke_mrici", "cortical_aca_mril",
               "cortical_mca_mril", "subcortical_aca_mril", "subcortical_mca_mril", "pca_cortex_mril",
               "thalamus_mril", "brainstem_mril", "cerebellum_mril", "watershed_mril", "hemorrhagic_infarct_mril",
               "old_stroke_mrich"]

    continuous = ["height_nm", "weight_nm", "sbp_nm", "dbp_nm", "bt_nm", "hr_nm", "rr_nm", "hb_nm",
                  "hct_nm", "platelet_nm", "wbc_nm", "ptt1_nm", "ptt2_nm", "ptinr_nm", "er_nm", "bun_nm",
                  "cre_nm", "ua_nm", "tcho_nm", "tg_nm", "hdl_nm", "ldl_nm", "gpt_nm", "age", "hospitalised_time"]

    barthel = ["feeding", "transfers", "bathing", "toilet_use", "grooming", "mobility", "stairs", "dressing",
               "bowel_control", "bladder_control"]

    nihss_in = ["nihs_1a_in", "nihs_1b_in", "nihs_1c_in", "nihs_2_in", "nihs_3_in", "nihs_4_in", "nihs_5al_in",
                "nihs_5br_in", "nihs_6al_in", "nihs_6br_in", "nihs_7_in", "nihs_8_in", "nihs_9_in", "nihs_10_in",
                "nihs_11_in"]

    nihss_out = ["nihs_1a_out", "nihs_1b_out", "nihs_1c_out", "nihs_2_out", "nihs_3_out",
                 "nihs_4_out", "nihs_5al_out", "nihs_5br_out", "nihs_6al_out", "nihs_6br_out", "nihs_7_out",
                 "nihs_8_out", "nihs_9_out", "nihs_10_out", "nihs_11_out"]

#     csv_path = os.path.join("..", "data", "tsr_train_all.csv")
#     tsr_all_df = pd.read_csv(csv_path)
#     print(tsr_all_df.shape)
#     tsr_all_df_1 = preprocessing(tsr_all_df)
#     csv_save = os.path.join("..", "data", "tsr_train_all_cleaned.csv")
#     tsr_all_df_1.to_csv(csv_save, index=False)

#     csv_path = os.path.join("..", "data", "tsr_train_area1.csv")
#     tsr_all_train_n_df = pd.read_csv(csv_path)
#     print(tsr_all_train_n_df.shape)
#     tsr_all_train_n_df_1 = preprocessing(tsr_all_train_n_df)
#     csv_save = os.path.join("..", "data", "tsr_train_area1_cleaned.csv")
#     tsr_all_train_n_df_1.to_csv(csv_save, index=False)

    csv_path = os.path.join(".", "tsr_train_area2.csv")
    tsr_all_train_c_df = pd.read_csv(csv_path)
    print(tsr_all_train_c_df.shape)
    tsr_all_train_c_df_1 = preprocessing(tsr_all_train_c_df)
    csv_save = os.path.join(".", "tsr_train_area2_cleaned.csv")
    tsr_all_train_c_df_1.to_csv(csv_save, index=False)

#     csv_path = os.path.join("..", "data", "tsr_train_area3.csv")
#     tsr_all_train_s_df = pd.read_csv(csv_path)
#     print(tsr_all_train_s_df.shape)
#     tsr_all_train_s_df_1 = preprocessing(tsr_all_train_s_df)
#     csv_save = os.path.join("..", "data", "tsr_train_area3_cleaned.csv")
#     tsr_all_train_s_df_1.to_csv(csv_save, index=False)

    csv_path = os.path.join(".", "tsr_test_all.csv")
    tsr_all_test_df = pd.read_csv(csv_path)
    print(tsr_all_test_df.shape)
    tsr_all_test_df_1 = preprocessing(tsr_all_test_df)
    csv_save = os.path.join(".", "tsr_test_all_cleaned.csv")
    tsr_all_test_df_1.to_csv(csv_save, index=False)