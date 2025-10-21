# DrosoClimb - Drosophila Behavioral Analysis Pipeline

A comprehensive behavioral analysis pipeline for Drosophila optogenetic experiments using Chrimson2 activation. This package analyzes climbing behavior across light and dark phases to compute preference indices and effect sizes for MBON screening.

## Overview

This pipeline processes multi-driver behavioral data to extract and analyze climbing metrics during optogenetic stimulation. It computes deltaG values (Hedges' g effect sizes) for screening MBONs (Mushroom Body Output Neurons) behavioral phenotypes.

## Key Features

- **1-second minimum bout duration filtering**: Only movement periods ≥1 second are considered valid bouts
- **Preference index calculations**: Uses LAI-style formulas `(light - dark)/(light + dark)` for robust preference metrics
- **Multi-phase analysis**: Compares Dark, Full (light), and Recovery phases
- **Effect size computation**: Calculates Hedges' g with bootstrap confidence intervals
- **Automated processing**: Batch processing of multiple driver lines

## Experimental Design

**Phases:**
- **Dark Phase**: 20 seconds baseline without optogenetic stimulation
- **Full Phase**: 20 seconds with optogenetic stimulation (Chrimson2 activation)
- **Recovery Phase**: 20 seconds post-stimulation recovery

**Comparison Groups:**
- **Experimental (Expt)**: Driver × Chrimson2 flies
- **Control (WT)**: w1118 × Chrimson2 flies

## Behavioral Metrics

### 1. **Fall Events**
- **Description**: Binary detection of falling events (Y-position drops >4.94mm)
- **Metrics**:
  - `fallnumber_meandiff`: Mean difference in fall counts between conditions
  - `fallnumber_bootstrap`: Bootstrap samples for statistical testing

### 2. **Movement Speed**
- **Description**: Overall velocity during movement periods
- **Metrics**:
  - `speed_deltag`: Hedges' g effect size for speed differences
  - `speed_bootstrap`: Bootstrap samples
- **Calculation**: Euclidean distance between consecutive frames divided by time interval

### 3. **Height/Position**
- **Description**: Average Y-position (climbing height) during each phase
- **Metrics**:
  - `height_deltag`: Effect size for height preferences
  - `height_bootstrap`: Bootstrap samples

### 4. **Bout Speed (BSpeed)**
- **Description**: Speed during valid movement bouts only (≥1 second duration)
- **Metrics**:
  - `bspeed_deltag`: Effect size for bout-specific speeds
  - `bspeed_bootstrap`: Bootstrap samples
  - `bspeed_hedgesg`: Single group comparison effect size
- **Filtering**: Only includes speeds from movement periods lasting ≥1 second

### 5. **Pause Position**
- **Description**: Height during pausing periods
- **Metrics**:
  - `pausepos_deltag`: Effect size for pause location preferences
  - `pausepos_bootstrap`: Bootstrap samples

### 6. **Maximum Velocity**
- **Description**: Peak velocity achieved during each phase
- **Metrics**:
  - `maxvelocity_deltag`: Effect size for maximum speed differences
  - `maxvelocity_bootstrap`: Bootstrap samples
  - `maxvelocity_ratio_hedgesg`: Ratio comparison effect size

### 7. **Straightness Index**
- **Description**: Measure of movement path efficiency (straight vs. tortuous)
- **Metrics**:
  - `straightindex_deltag`: Effect size for movement straightness
  - `straightindex_bootstrap`: Bootstrap samples
- **Calculation**: Ratio of displacement to total distance traveled

### 8. **Mean Bout Duration**
- **Description**: Average duration of movement bouts (≥1 second filter applied)
- **Metrics**:
  - `meanbout_deltag`: Effect size for bout duration differences  
  - `meanbout_bootstrap`: Bootstrap samples
  - `boutduration_ratio_hedgesg`: Ratio comparison effect size

### 9. **Bout Count/Number**
- **Description**: Number of discrete movement bouts per phase (≥1 second filter applied)
- **Metrics**:
  - `bout_deltag`: Effect size for bout frequency differences
  - `bout_bootstrap`: Bootstrap samples

### 10. **Bout Number Index** ⭐ **NEW**
- **Description**: Preference index for bout frequency between light vs dark phases
- **Formula**: `(bouts_light - bouts_dark) / (bouts_light + bouts_dark)`
- **Metrics**:
  - `boutnumber_index_hedgesg`: Effect size for bout preference
  - `boutnumber_index_bootstrap`: Bootstrap samples
- **Range**: -1 (dark preference) to +1 (light preference)
- **Advantages**: No division-by-zero issues, consistent with other preference indices

## Key Functions

### Core Analysis Functions (NLMATH.py)

#### Bout Detection & Filtering
- `countval()`: Counts consecutive events with 1-second minimum duration filter
- `boutspeed()`: Extracts velocities from valid bout periods only
- `behavior()`: Analyzes pause and bout patterns
- `boutanalysis()`: Comprehensive bout statistics per phase

#### Preference Index Calculations
- `boutindex()`: **NEW** - Calculates bout preference index using LAI formula
- `simplemetricratio()`: Calculates simple ratios for duration and velocity metrics
- `log2metric()`: Log2 ratio transformations

#### Statistical Analysis
- `deltaversion_deltag()`: Hedges' g effect size calculations with bootstrap
- `deltaversion_meandiff()`: Mean difference calculations for binary metrics
- `singledelta()`: Single group comparisons

#### Data Processing
- `pausecomp()`: Comprehensive pause/bout analysis across phases
- `velodabest()`: Speed analysis with proper NaN handling
- `maxvelocity()`: Peak velocity extraction per phase

### Preprocessing Functions (NLCLIMB.py)
- `generation()`: Raw data processing and quality control
- `timerule()`: Time window selection and phase separation
- `speedcalc()`: Frame-by-frame velocity calculations

## Recent Updates

### Version 2025.01
1. **Enhanced Bout Detection**: 
   - Implemented 1-second minimum duration filter for all bout-related metrics
   - Prevents spurious short movements from being classified as bouts

2. **Bout Index Implementation**:
   - Replaced bout ratio with bout index using preference formula
   - Eliminates division-by-zero issues when flies have no bouts in one condition
   - Provides interpretable -1 to +1 scale consistent with other preference indices

3. **Improved Velocity Filtering**:
   - `boutspeed()` function now applies duration filtering consistently
   - Only includes velocities from movement periods ≥1 second

## Output Files

**File Format**: `{Driver} x Chrimson2_deltag_allstats.csv`

**Columns**: Each metric includes both effect size and bootstrap columns for comprehensive statistical analysis.

## Dependencies

- pandas
- numpy  
- dabest (for effect size calculations)
- itertools (for bout detection)
- scipy (for statistical functions)

## Usage

```python
# Main processing script
exec(open('3. Multi-file processing.ipynb').read())
```

The pipeline automatically processes all driver lines in the specified directory and outputs comprehensive behavioral statistics for downstream analysis.

## Citation

If you use this pipeline, please cite the appropriate publications for:
- DABEST package for effect size calculations
- Chrimson2 optogenetic system
- Original behavioral analysis methodology