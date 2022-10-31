#!/bin/bash


# Set the base dir
BASE_DIR="../generated_problems/analysis/"

run_experiment ()
{
    echo "####" $1
    rm problems/*
    cp -a "$BASE_DIR$1/." problems/
    setup-experiment -i -f experiment_config -s vip-hlp-evaluate -q
}

#run_experiment "sine_1_1"
#run_experiment "output_clausified_caption_axiom_remapping_sine_1_0"


run_experiment "output_clausified_caption_axiom_remapping_sine_1_0"
run_experiment "output_clausified_caption_axiom_remapping_sine_1_1"
run_experiment "output_clausified_caption_axiom_remapping_sine_1_2"
run_experiment "output_clausified_caption_axiom_remapping_sine_1_4"
run_experiment "output_clausified_caption_axiom_remapping_sine_1.5_0"
run_experiment "output_clausified_caption_axiom_remapping_sine_1.5_1"
run_experiment "output_clausified_caption_axiom_remapping_sine_1.5_2"
run_experiment "output_clausified_caption_axiom_remapping_sine_1.5_4"
run_experiment "output_clausified_caption_axiom_remapping_sine_1.5_8"
run_experiment "output_clausified_caption_axiom_remapping_sine_1_8"
run_experiment "output_clausified_caption_axiom_remapping_sine_2_0"
run_experiment "output_clausified_caption_axiom_remapping_sine_2_1"
run_experiment "output_clausified_caption_axiom_remapping_sine_2_2"
run_experiment "output_clausified_caption_axiom_remapping_sine_2_4"
run_experiment "output_clausified_caption_axiom_remapping_sine_2.5_0"
run_experiment "output_clausified_caption_axiom_remapping_sine_2.5_1"
run_experiment "output_clausified_caption_axiom_remapping_sine_2.5_2"
run_experiment "output_clausified_caption_axiom_remapping_sine_2.5_4"
run_experiment "output_clausified_caption_axiom_remapping_sine_2.5_8"
run_experiment "output_clausified_caption_axiom_remapping_sine_2_8"



echo "# Finished #"

