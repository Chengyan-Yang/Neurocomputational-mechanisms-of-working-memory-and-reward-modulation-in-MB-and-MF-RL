script_frame <- Filter(function(f) !is.null(f$ofile), sys.frames())
if (length(script_frame)) {
  frame <- script_frame[[length(script_frame)]]
  script_path <- frame$ofile
  if (!is.null(frame$owd) && !grepl("^(/|[A-Za-z]:|\\\\\\\\)", script_path)) {
    script_path <- file.path(frame$owd, script_path)
  }
} else {
  script_path <- sub("^--file=", "", grep("^--file=", commandArgs(FALSE), value = TRUE)[1L])
}
here <- dirname(normalizePath(script_path, mustWork = TRUE))
project_dir <- dirname(dirname(dirname(here)))
# This script preprocesses raw behavioral data from the two-stage reinforcement learning task in the MEG experiment 
# to generate standardized datasets for subsequent reinforcement learning (RL) model fitting.
# The main procedures include data selection and cleaning, valid trial extraction, stimulus and choice coding, second-stage 
# state and reward information processing, and calculation of reaction times and behavioral performance measures. 
# Information on working memory load, reward magnitude, and probe trials is also retained. 
# The script generates a separate CSV file for each participant for subsequent computational modeling and behavioral analyses.

library(dplyr)
library(abind)
library(readxl)
library(tidyr)

# load data
folder_path <- file.path(project_dir, "data", "BHV", "2_stage_task", "original")
learning_dir <- file.path(dirname(folder_path), "preprocessed", "learning")
dir.create(file.path(learning_dir, "all", "RL_code"), recursive = TRUE, showWarnings = FALSE)
dir.create(file.path(learning_dir, "low_load"), recursive = TRUE, showWarnings = FALSE)
dir.create(file.path(learning_dir, "high_load"), recursive = TRUE, showWarnings = FALSE)
raw_data_files <- list.files(path = folder_path, pattern = "*.csv", full.names = TRUE)

cr_lload_small_magnitude <- numeric(length(raw_data_files))
cr_lload_large_magnitude <- numeric(length(raw_data_files))
cr_hload_small_magnitude <- numeric(length(raw_data_files))
cr_hload_large_magnitude <- numeric(length(raw_data_files))
subid <- numeric(length(raw_data_files))


for (n in seq_along(raw_data_files)) {
  raw_data <- read.csv(raw_data_files[n], header = TRUE, fileEncoding = "UTF-8-BOM") 
  
  # 1. Select values to process
  select_data <- raw_data[c('participant','date', # basic info
                            'block.thisN','nTrial','pair_2.thisN','pair_4.thisN', # trials info
                            'sequence','stake_image','stake5', # sequence and magnitude info
                            'imageFile','stage1_resp.keys','stage1_resp.rt','stage1_resp_2.keys','stage1_resp_2.rt', # BHV in the first(choosing) stage
                            'imageFile2','space.keys','space.rt','space_2.keys','space_2.rt', # BHV in the second(collecting) stage
                            'doprobe_brown','doprobe_black','doprobe_grey','doprobe_purple', # probe trials, brown(white), black(blue)
                            'n','n_white','n_blue','n_total_2','n_brown','n_black','n_grey','n_purple','n_total_4', # reward info
                            'interval_rest4.thisN','interval_rest_2.thisN' # rest info
                            )]
  
  
  # 2. Remove missing and null values; the remaining data are valid trials.
  data <- select_data %>%
    # Convert empty strings to NA
    mutate(across(where(is.character), ~ {
      ifelse(. == "", NA_character_, .)
    })) %>%
    group_by(pair_2.thisN, pair_4.thisN) %>%
    summarise(across(everything(), ~ coalesce(.[1], .[2]))) %>%
    ungroup()
  
  data <- data %>%
    filter(!(is.na(space.rt) & is.na(space_2.rt)) & !(space.rt == '' & space_2.rt == '')) %>%
    mutate(
      miss = 0,
      nTrial_globe = ifelse(block.thisN==0, nTrial, nTrial + 160),
      probe = if_else(if_any(starts_with("doprobe"), ~ coalesce(., 0) != 0), 1, 0)
    ) %>%
    arrange(block.thisN, pair_2.thisN, pair_4.thisN)
  
  
  # 3. Locate and log the missing trials
  ## Determine the maximum trial number
  max_trial_2 <- 159
  max_trial_4 <- 159
  ## Identify the missing trials and the probe trials, Label in the data
  missing_trials_2 <- setdiff(0:max_trial_2, data$pair_2.thisN)+1
  missing_trials_4 <- setdiff(0:max_trial_4, data$pair_4.thisN)+1

  # 5. Data Transformation
  all_trials = length(data$imageFile) 
  ## Initialize the vectors
  participant <- numeric(all_trials)
  pairs <- numeric(all_trials)
  stimuli_left <- numeric(all_trials)
  stimuli_right <- numeric(all_trials)
  choice <- numeric(all_trials)
  animal <- numeric(all_trials)
  rt_1 <- numeric(all_trials)
  rt_2 <- numeric(all_trials)
  rews_1 <- numeric(all_trials)
  rews_2 <- numeric(all_trials)
  rews_3 <- numeric(all_trials)
  rews_4 <- numeric(all_trials)
  magnitude <- numeric(all_trials)
  load <- numeric(all_trials)
  mean_rews <- numeric(all_trials)
  max_rews <- numeric(all_trials)
  score <- numeric(all_trials)
  points <- numeric(all_trials)
  part <- numeric(all_trials)
  trials <- numeric(all_trials)
  correct <- numeric(all_trials)
  
  ## Extract values trial by trial
  for(i in 1:all_trials){
    ### Encode the presented doors(low load condition)/boats(high load condition), 
    ### where 'pairs' represents the pair number (1,2/ 1,2,3,4), 'stimuli' represents left/right, 
    ### and 'choice' represents the door number(1,2,3,4) or boat number(1,2,3,4,5,6,7,8).
    
    #### Low load - first(choosing) stage
    if(data$imageFile[i] == 'image\\pair2\\stage1_1.PNG'){
      pairs[i] <- 1
      stimuli_left[i] <- 1
      stimuli_right[i] <- 2
      load[i] <- 'low'
      part[i] <- 0
      if(data$stage1_resp.keys[i] == "2"){
        choice[i] <- 1
      } else{
        choice[i] <- 2
      }
    } else if(data$imageFile[i] == 'image\\pair2\\stage1_11.PNG'){
      pairs[i] <- 1
      stimuli_left[i] <- 2
      stimuli_right[i] <- 1
      load[i] <- 'low'
      part[i] <- 0
      if(data$stage1_resp.keys[i] == "2"){
        choice[i] <- 2
      } else{
        choice[i] <- 1
      }
    } else if(data$imageFile[i] == 'image\\pair2\\stage1_2.PNG'){
      pairs[i] <- 2
      stimuli_left[i] <- 3
      stimuli_right[i] <- 4
      load[i] <- 'low'
      part[i] <- 0
      if(data$stage1_resp.keys[i] == "2"){
        choice[i] <- 3
      } else{
        choice[i] <- 4
      }
    } else if(data$imageFile[i] == 'image\\pair2\\stage1_21.PNG'){
      pairs[i] <- 2
      stimuli_left[i] <- 4
      stimuli_right[i] <- 3
      load[i] <- 'low'
      part[i] <- 0
      if(data$stage1_resp.keys[i] == "2"){
        choice[i] <- 4
      } else{
        choice[i] <- 3
      }
    #### High load - first(choosing) stage - part1
    } else if(data$imageFile[i] == 'image\\pair4\\stage1_1.PNG'){
      pairs[i] <- 1
      stimuli_left[i] <- 1
      stimuli_right[i] <- 2
      load[i] <- 'high'
      part[i] <- 1
      if(data$stage1_resp_2.keys[i] == "2"){
        choice[i] <- 1
      } else{
        choice[i] <- 2
      }
    } else if(data$imageFile[i] == 'image\\pair4\\stage1_11.PNG'){
      pairs[i] <- 1
      stimuli_left[i] <- 2
      stimuli_right[i] <- 1
      load[i] <- 'high'
      part[i] <- 1
      if(data$stage1_resp_2.keys[i] == "2"){
        choice[i] <- 2
      } else{
        choice[i] <- 1
      }
    } else if(data$imageFile[i] == 'image\\pair4\\stage1_2.PNG'){
      pairs[i] <- 2
      stimuli_left[i] <- 3
      stimuli_right[i] <- 4
      load[i] <- 'high'
      part[i] <- 1
      if(data$stage1_resp_2.keys[i] == "2"){
        choice[i] <- 3
      } else{
        choice[i] <- 4
      }
    } else if(data$imageFile[i] == 'image\\pair4\\stage1_21.PNG'){
      pairs[i] <- 2
      stimuli_left[i] <- 4
      stimuli_right[i] <- 3
      load[i] <- 'high'
      part[i] <- 1
      if(data$stage1_resp_2.keys[i] == "2"){
        choice[i] <- 4
      } else{
        choice[i] <- 3
      }
      #### High load - first(choosing) stage - part2
    } else if(data$imageFile[i] == 'image\\pair4\\stage1_3.PNG'){
      pairs[i] <- 3
      stimuli_left[i] <- 5
      stimuli_right[i] <- 6
      load[i] <- 'high'
      part[i] <- 2
      if(data$stage1_resp_2.keys[i] == "2"){
        choice[i] <- 5
      } else{
        choice[i] <- 6
      }
    } else if(data$imageFile[i] == 'image\\pair4\\stage1_31.PNG'){
      pairs[i] <- 3
      stimuli_left[i] <- 6
      stimuli_right[i] <- 5
      load[i] <- 'high'
      part[i] <- 2
      if(data$stage1_resp_2.keys[i] == "2"){
        choice[i] <- 6
      } else{
        choice[i] <- 5
      }
    } else if(data$imageFile[i] == 'image\\pair4\\stage1_4.PNG'){
      pairs[i] <- 4
      stimuli_left[i] <- 7
      stimuli_right[i] <- 8
      load[i] <- 'high'
      part[i] <- 2
      if(data$stage1_resp_2.keys[i] == "2"){
        choice[i] <- 7
      } else{
        choice[i] <- 8
      }
    } else if(data$imageFile[i] == 'image\\pair4\\stage1_41.PNG'){
      pairs[i] <- 4
      stimuli_left[i] <- 8
      stimuli_right[i] <- 7
      load[i] <- 'high'
      part[i] <- 2
      if(data$stage1_resp_2.keys[i] == "2"){
        choice[i] <- 8
      } else{
        choice[i] <- 7
      }
    }
    
    ### Encode the presented animal
    if(data$imageFile2[i] == 'image\\pair2\\stage2_white.PNG'){
      animal[i] <- 1
    } else if(data$imageFile2[i] == 'image\\pair2\\stage2_blue.PNG'){
      animal[i] <- 2
    } else if(data$imageFile2[i] == 'image\\pair4\\stage2_brown.PNG'){
      animal[i] <- 1
    } else if(data$imageFile2[i] == 'image\\pair4\\stage2_black.PNG'){
      animal[i] <- 2
    } else if(data$imageFile2[i] == 'image\\pair4\\stage2_grey.PNG'){
      animal[i] <- 3
    } else if(data$imageFile2[i] == 'image\\pair4\\stage2_purple.PNG'){
      animal[i] <- 4
    } else{
      animal[i] <- 0
    }
    
    ### Encode reward magnitude
    if(data$stake5[i] == 0){
      magnitude[i] <- 'small'
      points[i] <- data$n[i]
    } else if(data$stake5[i] == 1){
      magnitude[i] <- 'large'
      points[i] <- data$n[i] / 5
    }
    
    ### Value extraction and computation
    if(load[i]=='low'){
      trials[i] <- data$pair_2.thisN[i] + 1
      rt_1[i] <- data$stage1_resp.rt[i] 
      rt_2[i] <- data$space.rt[i]
      rews_1[i] <- data$n_white[i] 
      rews_2[i] <- data$n_blue[i] 
      score[i] <- data$n_total_2[i]
      max_rews[i] <- max(rews_1[i], rews_2[i])
      if(points[i]==max_rews[i]){
        correct[i] <- 1
      }else{
        correct[i] <- 0
      }
    } else if(load[i]=='high'){
      trials[i] <- data$pair_4.thisN[i] + 1
      rt_1[i] <- data$stage1_resp_2.rt[i] 
      rt_2[i] <- data$space_2.rt[i]
      score[i] <- data$n_total_4[i]
      rews_1[i] <- data$n_brown[i]
      rews_2[i] <- data$n_black[i]
      rews_3[i] <- data$n_grey[i]
      rews_4[i] <- data$n_purple[i]
      if(part[i]==1){
        max_rews[i] <- max(rews_1[i], rews_2[i])
        if(points[i]==max_rews[i]){
          correct[i] <- 1
        }else{
          correct[i] <- 0
        }
      } else if(part[i]==2){
        max_rews[i] <- max(rews_3[i], rews_4[i])
        if(points[i]==max_rews[i]){
          correct[i] <- 1
        }else{
          correct[i] <- 0
        }
      }
    }
    participant[i] <- data$participant[i]
  }
  
  # 6. Merge all the cleaned and organized data
  df <- data.frame(participant = data$participant, trials = as.numeric(trials), trials_globe = as.numeric(data$nTrial_globe),block = as.numeric(data$block.thisN + 1),
                   pairs = as.numeric(pairs), stimuli_left = as.numeric(stimuli_left), stimuli_right = as.numeric(stimuli_right),
                   choice = as.numeric(choice), animal = as.numeric(animal), rt_1 = as.numeric(rt_1), rt_2 = as.numeric(rt_2), 
                   rews_1 = as.numeric(rews_1), rews_2 = as.numeric(rews_2),rews_3 = as.numeric(rews_3), rews_4 = as.numeric(rews_4), max_rews = as.numeric(max_rews),
                   magnitude = magnitude, load = load, part = as.numeric(part), score = as.numeric(score), points = as.numeric(points),
                   correct = correct, miss = data$miss, probe = data$probe, stringsAsFactors = FALSE)
  
  # 7. Data grouping and reward acquisition rate calculation
  ## Low load
  data_lowload <- df[df$part==0,]
  data_lowload_smallmagnitude <- data_lowload[data_lowload$magnitude=='small',]
  data_lowload_largemagnitude <- data_lowload[data_lowload$magnitude=='large',]
  cr_lowload_smallmagnitude <- sum(as.numeric(data_lowload_smallmagnitude$points[data_lowload_smallmagnitude$probe!=1])) / sum(as.numeric(data_lowload_smallmagnitude$max_rews[data_lowload_smallmagnitude$probe!=1]))
  cr_lowload_largemagnitude <- sum(as.numeric(data_lowload_largemagnitude$points[data_lowload_largemagnitude$probe!=1])) / sum(as.numeric(data_lowload_largemagnitude$max_rews[data_lowload_largemagnitude$probe!=1]))
  data_lowload <- data_lowload %>%
    complete(trials = 1:160, fill = list(miss = 1))
  
  ## High load
  data_highload <- df[df$part==1 | df$part == 2,]
  data_highload_smallmagnitude <- data_highload[data_highload$magnitude=='small',]
  data_highload_largemagnitude <- data_highload[data_highload$magnitude=='large',]
  cr_highload_smallmagnitude <- sum(as.numeric(data_highload_smallmagnitude$points[data_highload_smallmagnitude$probe!=1])) / sum(as.numeric(data_highload_smallmagnitude$max_rews[data_highload_smallmagnitude$probe!=1]))
  cr_highload_largemagnitude <- sum(as.numeric(data_highload_largemagnitude$points[data_highload_largemagnitude$probe!=1])) / sum(as.numeric(data_highload_largemagnitude$max_rews[data_highload_largemagnitude$probe!=1]))
  data_highload <- data_highload %>%
    complete(trials = 1:160, fill = list(miss = 1))
  
  
  # 8. Summarize participant-level metrics
  cr_lload_small_magnitude[n] <- cr_lowload_smallmagnitude
  cr_lload_large_magnitude[n] <- cr_lowload_largemagnitude
  cr_hload_small_magnitude[n] <- cr_highload_smallmagnitude
  cr_hload_large_magnitude[n] <- cr_highload_largemagnitude
  participant_id <- unique(na.omit(raw_data$participant))
  if (length(participant_id) != 1L || !participant_id %in% 1:52) {
    stop("Expected one participant ID from 1 to 52 in ", raw_data_files[n])
  }
  subid[n] <- participant_id
  
  # 9. output
  df<- df %>%
    complete(trials_globe = 1:320, fill = list(miss = 1))
  output_file <- file.path(file.path(learning_dir, "all", "RL_code"), 
                                   gsub(".csv", "_prep.csv", basename(raw_data_files[n])))
  write.csv(df, output_file, row.names = FALSE)
  print(paste("Processed and saved:", output_file))
  
  output_file_lowload <- file.path(file.path(learning_dir, "low_load"), 
                                   gsub(".csv", "_lload.csv", basename(raw_data_files[n])))
  write.csv(data_lowload, output_file_lowload, row.names = FALSE)
  print(paste("Processed and saved:", output_file_lowload))
  
  output_file_highload <- file.path(file.path(learning_dir, "high_load"), 
                                    gsub(".csv", "_hload.csv", basename(raw_data_files[n])))
  write.csv(data_highload, output_file_highload, row.names = FALSE)
  print(paste("Processed and saved:", output_file_highload))
}

# 10. participant-level metrics output
stopifnot(length(subid) == 52L, setequal(subid, 1:52))
data_overall <- data.frame(
  participant = subid,
  cr_lload_small_magnitude = cr_lload_small_magnitude,
  cr_lload_large_magnitude = cr_lload_large_magnitude,
  cr_hload_small_magnitude = cr_hload_small_magnitude,
  cr_hload_large_magnitude = cr_hload_large_magnitude
)
output_file <- file.path(learning_dir, 
                         "performance_overall.csv") 
write.csv(data_overall, output_file, row.names = FALSE)
print(paste("Processed and saved:", output_file))
