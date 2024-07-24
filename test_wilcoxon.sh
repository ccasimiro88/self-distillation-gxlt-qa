#!/bin/bash
script_dir="$(cd "$(dirname "$0")" && pwd)"

# SKD vs CE
model_dir_a=$script_dir/runs/joint_train-xquad/en-es-de-ar-vi-hi-zh/mbert-qa-en/ep-3/ntl-3/ce/seed-3
model_dir_b=$script_dir/runs/joint_train-xquad/en-es-de-ar-vi-hi-zh/mbert-qa-en/ep-3/ntl-5/skd/temp-2/seed-3/seed-3/

results_file=$model_dir_b/test_wilcoxon_results_skd_vs_ce
if [ ! -f $results_file ]; then
    echo "" | tee $results_file

    # MLQA-dev
    # G-XLT
    testset=mlqa-dev
    task=G-XLT
    model_a=ce
    model_b=skd
    file_a=$model_dir_a/eval_results_example_level_${testset}_${model_a}
    file_b=$model_dir_b/eval_results_example_level_${testset}_${model_b}


    echo "Testset: ${testset}" | tee -a $results_file
    echo "Task: ${task}" | tee -a $results_file
    python $script_dir/src/wilcoxon_test.py --file_a $file_a --file_b $file_b --do_gxlt | tee -a $results_file
    echo "" | tee -a $results_file

    # XLT
    testset=mlqa-dev
    task=XLT
    model_a=ce
    model_b=skd
    file_a=$model_dir_a/eval_results_example_level_${testset}_${model_a}
    file_b=$model_dir_b/eval_results_example_level_${testset}_${model_b}

    langs="en es de ar hi vi zh"


    echo "Testset: ${testset}" | tee -a $results_file
    echo "Task: ${task}" | tee -a $results_file
    python $script_dir/src/wilcoxon_test.py --file_a $file_a --file_b $file_b | tee -a $results_file
    echo "" | tee -a $results_file

    # XLT
    testset=mlqa-dev
    task=XLT
    model_a=ce
    model_b=skd
    file_a=$model_dir_a/eval_results_example_level_${testset}_${model_a}
    file_b=$model_dir_b/eval_results_example_level_${testset}_${model_b}

    for lang in $langs; do
        langs="en es de ar hi vi zh"


        echo "Testset: ${testset}" | tee -a $results_file
        echo "Task: ${task}" | tee -a $results_file
        echo "Languages: ${lang}" | tee -a $results_file
        python $script_dir/src/wilcoxon_test.py --file_a $file_a --file_b $file_b --languages $lang | tee -a $results_file
        echo "" | tee -a $results_file
    done

    # MLQA-test
    # G-XLT
    testset=mlqa-test
    task=G-XLT
    model_a=ce
    model_b=skd
    file_a=$model_dir_a/eval_results_example_level_${testset}_${model_a}
    file_b=$model_dir_b/eval_results_example_level_${testset}_${model_b}


    echo "Testset: ${testset}" | tee -a $results_file
    echo "Task: ${task}" | tee -a $results_file
    python $script_dir/src/wilcoxon_test.py --file_a $file_a --file_b $file_b --do_gxlt | tee -a $results_file
    echo "" | tee -a $results_file

    # XLT
    testset=mlqa-test
    task=XLT
    model_a=ce
    model_b=skd
    file_a=$model_dir_a/eval_results_example_level_${testset}_${model_a}
    file_b=$model_dir_b/eval_results_example_level_${testset}_${model_b}

    langs="en es de ar hi vi zh"


    echo "Testset: ${testset}" | tee -a $results_file
    echo "Task: ${task}" | tee -a $results_file
    python $script_dir/src/wilcoxon_test.py --file_a $file_a --file_b $file_b | tee -a $results_file
    echo "" | tee -a $results_file

    # XLT - single language
    testset=mlqa-test
    task=XLT
    model_a=ce
    model_b=skd
    file_a=$model_dir_a/eval_results_example_level_${testset}_${model_a}
    file_b=$model_dir_b/eval_results_example_level_${testset}_${model_b}

    for lang in $langs; do
        langs="en es de ar hi vi zh"


        echo "Testset: ${testset}" | tee -a $results_file
        echo "Task: ${task}" | tee -a $results_file
        echo "Languages: ${lang}" | tee -a $results_file
        python $script_dir/src/wilcoxon_test.py --file_a $file_a --file_b $file_b --languages $lang | tee -a $results_file
        echo "" | tee -a $results_file
    done
fi

# SKD_MAP vs CE
model_dir_a=$script_dir/runs/joint_train-xquad/en-es-de-ar-vi-hi-zh/mbert-qa-en/ep-3/ntl-3/ce/seed-3
model_dir_b=$script_dir/runs/joint_train-xquad/en-es-de-ar-vi-hi-zh/mbert-qa-en/ep-3/ntl-5/skd_map/temp-2/seed-3/seed-3

results_file=$model_dir_b/test_wilcoxon_results_skd_map_vs_ce
if [ ! -f $results_file ]; then
    echo "" | tee $results_file

    # MLQA-dev
    # G-XLT
    testset=mlqa-dev
    task=G-XLT
    model_a=ce
    model_b=skd_map
    file_a=$model_dir_a/eval_results_example_level_${testset}_${model_a}
    file_b=$model_dir_b/eval_results_example_level_${testset}_${model_b}


    echo "Testset: ${testset}" | tee -a $results_file
    echo "Task: ${task}" | tee -a $results_file
    python $script_dir/src/wilcoxon_test.py --file_a $file_a --file_b $file_b --do_gxlt | tee -a $results_file
    echo "" | tee -a $results_file

    # XLT
    testset=mlqa-dev
    task=XLT
    model_a=ce
    model_b=skd_map
    file_a=$model_dir_a/eval_results_example_level_${testset}_${model_a}
    file_b=$model_dir_b/eval_results_example_level_${testset}_${model_b}

    langs="en es de ar hi vi zh"


    echo "Testset: ${testset}" | tee -a $results_file
    echo "Task: ${task}" | tee -a $results_file
    python $script_dir/src/wilcoxon_test.py --file_a $file_a --file_b $file_b | tee -a $results_file
    echo "" | tee -a $results_file

    # XLT
    testset=mlqa-dev
    task=XLT
    model_a=ce
    model_b=skd_map
    file_a=$model_dir_a/eval_results_example_level_${testset}_${model_a}
    file_b=$model_dir_b/eval_results_example_level_${testset}_${model_b}

    for lang in $langs; do
        langs="en es de ar hi vi zh"


        echo "Testset: ${testset}" | tee -a $results_file
        echo "Task: ${task}" | tee -a $results_file
        echo "Languages: ${lang}" | tee -a $results_file
        python $script_dir/src/wilcoxon_test.py --file_a $file_a --file_b $file_b --languages $lang | tee -a $results_file
        echo "" | tee -a $results_file
    done

    # MLQA-test
    # G-XLT
    testset=mlqa-test
    task=G-XLT
    model_a=ce
    model_b=skd_map
    file_a=$model_dir_a/eval_results_example_level_${testset}_${model_a}
    file_b=$model_dir_b/eval_results_example_level_${testset}_${model_b}


    echo "Testset: ${testset}" | tee -a $results_file
    echo "Task: ${task}" | tee -a $results_file
    python $script_dir/src/wilcoxon_test.py --file_a $file_a --file_b $file_b --do_gxlt | tee -a $results_file
    echo "" | tee -a $results_file

    # XLT
    testset=mlqa-test
    task=XLT
    model_a=ce
    model_b=skd_map
    file_a=$model_dir_a/eval_results_example_level_${testset}_${model_a}
    file_b=$model_dir_b/eval_results_example_level_${testset}_${model_b}

    langs="en es de ar hi vi zh"


    echo "Testset: ${testset}" | tee -a $results_file
    echo "Task: ${task}" | tee -a $results_file
    python $script_dir/src/wilcoxon_test.py --file_a $file_a --file_b $file_b | tee -a $results_file
    echo "" | tee -a $results_file

    # XLT - single language
    testset=mlqa-test
    task=XLT
    model_a=ce
    model_b=skd_map
    file_a=$model_dir_a/eval_results_example_level_${testset}_${model_a}
    file_b=$model_dir_b/eval_results_example_level_${testset}_${model_b}

    for lang in $langs; do
        langs="en es de ar hi vi zh"


        echo "Testset: ${testset}" | tee -a $results_file
        echo "Task: ${task}" | tee -a $results_file
        echo "Languages: ${lang}" | tee -a $results_file
        python $script_dir/src/wilcoxon_test.py --file_a $file_a --file_b $file_b --languages $lang | tee -a $results_file
        echo "" | tee -a $results_file
    done
fi


# SKD_MAP vs SKD
model_dir_a=$script_dir/runs/joint_train-xquad/en-es-de-ar-vi-hi-zh/mbert-qa-en/ep-3/ntl-5/skd/temp-2/seed-3/seed-3
model_dir_b=$script_dir/runs/joint_train-xquad/en-es-de-ar-vi-hi-zh/mbert-qa-en/ep-3/ntl-5/skd_map/temp-2/seed-3/seed-3

results_file=$model_dir_b/test_wilcoxon_results_skd_map_vs_skd
if [ ! -f $results_file ]; then
    echo "" | tee $results_file

    # MLQA-dev
    # G-XLT
    testset=mlqa-dev
    task=G-XLT
    model_a=skd
    model_b=skd_map
    file_a=$model_dir_a/eval_results_example_level_${testset}_${model_a}
    file_b=$model_dir_b/eval_results_example_level_${testset}_${model_b}


    echo "Testset: ${testset}" | tee -a $results_file
    echo "Task: ${task}" | tee -a $results_file
    python $script_dir/src/wilcoxon_test.py --file_a $file_a --file_b $file_b --do_gxlt | tee -a $results_file
    echo "" | tee -a $results_file

    # XLT
    testset=mlqa-dev
    task=XLT
    model_a=skd
    model_b=skd_map
    file_a=$model_dir_a/eval_results_example_level_${testset}_${model_a}
    file_b=$model_dir_b/eval_results_example_level_${testset}_${model_b}

    langs="en es de ar hi vi zh"


    echo "Testset: ${testset}" | tee -a $results_file
    echo "Task: ${task}" | tee -a $results_file
    python $script_dir/src/wilcoxon_test.py --file_a $file_a --file_b $file_b | tee -a $results_file
    echo "" | tee -a $results_file

    # XLT
    testset=mlqa-dev
    task=XLT
    model_a=skd
    model_b=skd_map
    file_a=$model_dir_a/eval_results_example_level_${testset}_${model_a}
    file_b=$model_dir_b/eval_results_example_level_${testset}_${model_b}

    for lang in $langs; do
        langs="en es de ar hi vi zh"


        echo "Testset: ${testset}" | tee -a $results_file
        echo "Task: ${task}" | tee -a $results_file
        echo "Languages: ${lang}" | tee -a $results_file
        python $script_dir/src/wilcoxon_test.py --file_a $file_a --file_b $file_b --languages $lang | tee -a $results_file
        echo "" | tee -a $results_file
    done

    # MLQA-test
    # G-XLT
    testset=mlqa-test
    task=G-XLT
    model_a=skd
    model_b=skd_map
    file_a=$model_dir_a/eval_results_example_level_${testset}_${model_a}
    file_b=$model_dir_b/eval_results_example_level_${testset}_${model_b}


    echo "Testset: ${testset}" | tee -a $results_file
    echo "Task: ${task}" | tee -a $results_file
    python $script_dir/src/wilcoxon_test.py --file_a $file_a --file_b $file_b --do_gxlt | tee -a $results_file
    echo "" | tee -a $results_file

    # XLT
    testset=mlqa-test
    task=XLT
    model_a=skd
    model_b=skd_map
    file_a=$model_dir_a/eval_results_example_level_${testset}_${model_a}
    file_b=$model_dir_b/eval_results_example_level_${testset}_${model_b}

    langs="en es de ar hi vi zh"


    echo "Testset: ${testset}" | tee -a $results_file
    echo "Task: ${task}" | tee -a $results_file
    python $script_dir/src/wilcoxon_test.py --file_a $file_a --file_b $file_b | tee -a $results_file
    echo "" | tee -a $results_file

    # XLT - single language
    testset=mlqa-test
    task=XLT
    model_a=skd
    model_b=skd_map
    file_a=$model_dir_a/eval_results_example_level_${testset}_${model_a}
    file_b=$model_dir_b/eval_results_example_level_${testset}_${model_b}

    for lang in $langs; do
        langs="en es de ar hi vi zh"


        echo "Testset: ${testset}" | tee -a $results_file
        echo "Task: ${task}" | tee -a $results_file
        echo "Languages: ${lang}" | tee -a $results_file
        python $script_dir/src/wilcoxon_test.py --file_a $file_a --file_b $file_b --languages $lang | tee -a $results_file
        echo "" | tee -a $results_file
    done
fi