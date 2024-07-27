#!/bin/bash
script_dir="$(cd "$(dirname "$0")" && pwd)"

# SKD vs CE
model_dir_a=$script_dir/runs/joint_train-xquad/en-es-de-ar-vi-hi-zh/mbert-qa-en/ep-3/ntl-3/ce/seed-3
model_dir_b=$script_dir/runs/joint_train-xquad/en-es-de-ar-vi-hi-zh/mbert-qa-en/ep-3/ntl-5/ce-kl-fw-self-distil/temp-2/seed-3
model_a=ce
model_b=skd


results_file=$script_dir/runs/test_wilcoxon/results_${model_b}_vs_${model_a}
mkdir -p $script_dir/runs/test_wilcoxon/

if [ ! -f $results_file ]; then
    echo "" | tee $results_file

    # MLQA-test
    testset=mlqa-test
    file_a=$model_dir_a/eval_results_example_level_${testset}_${model_a}
    file_b=$model_dir_b/eval_results_example_level_${testset}_${model_b}

    # G-XLT
    task=G-XLT

    echo "Testset: ${testset}" | tee -a $results_file
    echo "Task: ${task}" | tee -a $results_file
    python $script_dir/src/wilcoxon_test.py --file_a $file_a --file_b $file_b --do_gxlt | tee -a $results_file
    echo "" | tee -a $results_file

    # XLT
    task=XLT

    echo "Testset: ${testset}" | tee -a $results_file
    echo "Task: ${task}" | tee -a $results_file
    python $script_dir/src/wilcoxon_test.py --file_a $file_a --file_b $file_b | tee -a $results_file
    echo "" | tee -a $results_file

    # XLT - single language
    langs="en es de ar hi vi zh"
    for lang in $langs; do


        echo "Testset: ${testset}" | tee -a $results_file
        echo "Task: ${task}" | tee -a $results_file
        echo "Languages: ${lang}" | tee -a $results_file
        python $script_dir/src/wilcoxon_test.py --file_a $file_a --file_b $file_b --languages $lang | tee -a $results_file
        echo "" | tee -a $results_file
    done

    # XQUAD
    # XLT
    testset=xquad
    file_a=$model_dir_a/eval_results_example_level_${testset}_${model_a}
    file_b=$model_dir_b/eval_results_example_level_${testset}_${model_b}
    task=XLT


    echo "Testset: ${testset}" | tee -a $results_file
    echo "Task: ${task}" | tee -a $results_file
    python $script_dir/src/wilcoxon_test.py --file_a $file_a --file_b $file_b | tee -a $results_file
    echo "" | tee -a $results_file

    # XLT - single language
    langs="el ru tr th"
    for lang in $langs; do

        echo "Testset: ${testset}" | tee -a $results_file
        echo "Task: ${task}" | tee -a $results_file
        echo "Languages: ${lang}" | tee -a $results_file
        python $script_dir/src/wilcoxon_test.py --file_a $file_a --file_b $file_b --languages $lang | tee -a $results_file
        echo "" | tee -a $results_file
    done

    # TyDiQA -goldp
    # XLT
    testset=tydiqa-goldp
    file_a=$model_dir_a/eval_results_example_level_${testset}_${model_a}
    file_b=$model_dir_b/eval_results_example_level_${testset}_${model_b}
    task=XLT

    echo "Testset: ${testset}" | tee -a $results_file
    echo "Task: ${task}" | tee -a $results_file
    python $script_dir/src/wilcoxon_test.py --file_a $file_a --file_b $file_b | tee -a $results_file
    echo "" | tee -a $results_file

    # XLT - single language
    langs="bengali finnish indonesian korean russian swahili telugu"
    for lang in $langs; do

        echo "Testset: ${testset}" | tee -a $results_file
        echo "Task: ${task}" | tee -a $results_file
        echo "Languages: ${lang}" | tee -a $results_file
        python $script_dir/src/wilcoxon_test.py --file_a $file_a --file_b $file_b --languages $lang | tee -a $results_file
        echo "" | tee -a $results_file
    done
fi

# SKD_MAP vs CE
model_dir_a=$script_dir/runs/joint_train-xquad/en-es-de-ar-vi-hi-zh/mbert-qa-en/ep-3/ntl-3/ce/seed-3
model_dir_b=$script_dir/runs/joint_train-xquad/en-es-de-ar-vi-hi-zh/mbert-qa-en/ep-3/ntl-5/ce-kl-fw-map-coeff-self-distil/temp-2/seed-3
model_a=ce
model_b=skd_map


results_file=$script_dir/runs/test_wilcoxon/results_${model_b}_vs_${model_a}
mkdir -p $script_dir/runs/test_wilcoxon/

if [ ! -f $results_file ]; then
    echo "" | tee $results_file

    # MLQA-test
    testset=mlqa-test
    file_a=$model_dir_a/eval_results_example_level_${testset}_${model_a}
    file_b=$model_dir_b/eval_results_example_level_${testset}_${model_b}
    # G-XLT
    task=G-XLT

    echo "Testset: ${testset}" | tee -a $results_file
    echo "Task: ${task}" | tee -a $results_file
    python $script_dir/src/wilcoxon_test.py --file_a $file_a --file_b $file_b --do_gxlt | tee -a $results_file
    echo "" | tee -a $results_file

    # XLT
    task=XLT

    echo "Testset: ${testset}" | tee -a $results_file
    echo "Task: ${task}" | tee -a $results_file
    python $script_dir/src/wilcoxon_test.py --file_a $file_a --file_b $file_b | tee -a $results_file
    echo "" | tee -a $results_file

    # XLT - single language
    langs="en es de ar hi vi zh"
    for lang in $langs; do


        echo "Testset: ${testset}" | tee -a $results_file
        echo "Task: ${task}" | tee -a $results_file
        echo "Languages: ${lang}" | tee -a $results_file
        python $script_dir/src/wilcoxon_test.py --file_a $file_a --file_b $file_b --languages $lang | tee -a $results_file
        echo "" | tee -a $results_file
    done

    # XQUAD
    # XLT
    testset=xquad
    file_a=$model_dir_a/eval_results_example_level_${testset}_${model_a}
    file_b=$model_dir_b/eval_results_example_level_${testset}_${model_b}
    task=XLT


    echo "Testset: ${testset}" | tee -a $results_file
    echo "Task: ${task}" | tee -a $results_file
    python $script_dir/src/wilcoxon_test.py --file_a $file_a --file_b $file_b | tee -a $results_file
    echo "" | tee -a $results_file

    # XLT - single language
    langs="el ru tr th"
    for lang in $langs; do

        echo "Testset: ${testset}" | tee -a $results_file
        echo "Task: ${task}" | tee -a $results_file
        echo "Languages: ${lang}" | tee -a $results_file
        python $script_dir/src/wilcoxon_test.py --file_a $file_a --file_b $file_b --languages $lang | tee -a $results_file
        echo "" | tee -a $results_file
    done

    # TyDiQA -goldp
    # XLT
    testset=tydiqa-goldp
    file_a=$model_dir_a/eval_results_example_level_${testset}_${model_a}
    file_b=$model_dir_b/eval_results_example_level_${testset}_${model_b}
    task=XLT

    echo "Testset: ${testset}" | tee -a $results_file
    echo "Task: ${task}" | tee -a $results_file
    python $script_dir/src/wilcoxon_test.py --file_a $file_a --file_b $file_b | tee -a $results_file
    echo "" | tee -a $results_file

    # XLT - single language
    langs="bengali finnish indonesian korean russian swahili telugu"
    for lang in $langs; do

        echo "Testset: ${testset}" | tee -a $results_file
        echo "Task: ${task}" | tee -a $results_file
        echo "Languages: ${lang}" | tee -a $results_file
        python $script_dir/src/wilcoxon_test.py --file_a $file_a --file_b $file_b --languages $lang | tee -a $results_file
        echo "" | tee -a $results_file
    done
fi

# SKD_MAP vs SKD
model_dir_a=$script_dir/runs/joint_train-xquad/en-es-de-ar-vi-hi-zh/mbert-qa-en/ep-3/ntl-5/ce-kl-fw-self-distil/temp-2/seed-3
model_dir_b=$script_dir/runs/joint_train-xquad/en-es-de-ar-vi-hi-zh/mbert-qa-en/ep-3/ntl-5/ce-kl-fw-map-coeff-self-distil/temp-2/seed-3
model_a=skd
model_b=skd_map


results_file=$script_dir/runs/test_wilcoxon/results_${model_b}_vs_${model_a}
mkdir -p $script_dir/runs/test_wilcoxon/

if [ ! -f $results_file ]; then
    echo "" | tee $results_file

    # MLQA-test
    testset=mlqa-test
    file_a=$model_dir_a/eval_results_example_level_${testset}_${model_a}
    file_b=$model_dir_b/eval_results_example_level_${testset}_${model_b}
    # G-XLT
    task=G-XLT

    echo "Testset: ${testset}" | tee -a $results_file
    echo "Task: ${task}" | tee -a $results_file
    python $script_dir/src/wilcoxon_test.py --file_a $file_a --file_b $file_b --do_gxlt | tee -a $results_file
    echo "" | tee -a $results_file

    # XLT
    task=XLT

    echo "Testset: ${testset}" | tee -a $results_file
    echo "Task: ${task}" | tee -a $results_file
    python $script_dir/src/wilcoxon_test.py --file_a $file_a --file_b $file_b | tee -a $results_file
    echo "" | tee -a $results_file

    # XLT - single language
    langs="en es de ar hi vi zh"
    for lang in $langs; do


        echo "Testset: ${testset}" | tee -a $results_file
        echo "Task: ${task}" | tee -a $results_file
        echo "Languages: ${lang}" | tee -a $results_file
        python $script_dir/src/wilcoxon_test.py --file_a $file_a --file_b $file_b --languages $lang | tee -a $results_file
        echo "" | tee -a $results_file
    done

    # XQUAD
    # XLT
    testset=xquad
    file_a=$model_dir_a/eval_results_example_level_${testset}_${model_a}
    file_b=$model_dir_b/eval_results_example_level_${testset}_${model_b}
    task=XLT


    echo "Testset: ${testset}" | tee -a $results_file
    echo "Task: ${task}" | tee -a $results_file
    python $script_dir/src/wilcoxon_test.py --file_a $file_a --file_b $file_b | tee -a $results_file
    echo "" | tee -a $results_file

    # XLT - single language
    langs="el ru tr th"
    for lang in $langs; do

        echo "Testset: ${testset}" | tee -a $results_file
        echo "Task: ${task}" | tee -a $results_file
        echo "Languages: ${lang}" | tee -a $results_file
        python $script_dir/src/wilcoxon_test.py --file_a $file_a --file_b $file_b --languages $lang | tee -a $results_file
        echo "" | tee -a $results_file
    done

    # TyDiQA -goldp
    # XLT
    testset=tydiqa-goldp
    file_a=$model_dir_a/eval_results_example_level_${testset}_${model_a}
    file_b=$model_dir_b/eval_results_example_level_${testset}_${model_b}
    task=XLT

    echo "Testset: ${testset}" | tee -a $results_file
    echo "Task: ${task}" | tee -a $results_file
    python $script_dir/src/wilcoxon_test.py --file_a $file_a --file_b $file_b | tee -a $results_file
    echo "" | tee -a $results_file

    # XLT - single language
    langs="bengali finnish indonesian korean russian swahili telugu"
    for lang in $langs; do

        echo "Testset: ${testset}" | tee -a $results_file
        echo "Task: ${task}" | tee -a $results_file
        echo "Languages: ${lang}" | tee -a $results_file
        python $script_dir/src/wilcoxon_test.py --file_a $file_a --file_b $file_b --languages $lang | tee -a $results_file
        echo "" | tee -a $results_file
    done
fi

