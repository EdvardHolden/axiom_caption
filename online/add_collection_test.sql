INSERT INTO Collection (CollectionName, CollectionDescription, Date)
        VALUES ("deepmath_test", "Some deepmath test problems", NOW());
   
   INSERT INTO CollectionProblems (Collection, Problem)
       SELECT LAST_INSERT_ID(),
       PV.ProblemVersionID
           FROM (Problem P INNER JOIN ProblemVersion PV ON P.ProblemID = PV.Problem)
               WHERE PV.Version = 64 AND P.ProblemName IN (
"l16_msuhom_1",
"t17_pdiff_3 ",
"t46_intpro_1",
"t12_circled1",
"t20_zf_model",
"t24_laplace ",
"t21_anproj_2",
"t36_equation",
"t52_quaterni",
"t55_intpro_1",
"l13_euclid_8",
"t25_rlsub_1 ",
"t82_bvfunc11",
"t80_matrix10",
"t31_vectsp_6",
"t7_matrixr1 ",
"t23_waybel_4",
"t67_rltopsp1",
"t31_group_1 ",
"t20_xxreal_3",
"t21_cat_4   ",
"t22_numbers ",
"t37_yellow_6",
"t26_complfld",
"t84_exchsort",
"t48_yellow_2",
"t5_integra1 ",
"l102_geomtra",
"l102_topreal",
"l10_algstr_1",
"l10_ami_wstd",
"l10_asympt_1",
"l10_ens_1   ",
"l10_flang_1 ",
"l10_fomodel0",
"l10_pnproc_1",
"l10_prvect_1"
);
