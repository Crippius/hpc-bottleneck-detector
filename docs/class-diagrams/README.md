# Class Diagram Views

This directory contains multiple views of the HPC Bottleneck Detector architecture, organized by concern for easier understanding.

## Overview

- **[0-full-view.mmd](0-full-view.mmd)** - Complete detailed class diagram with all components and relationships
- **[1-overview.mmd](1-overview.mmd)** - High-level system architecture showing main components and data flow

## Detailed Views

### 2. Data Layer View

**File:** [2-data-layer.mmd](2-data-layer.mmd)

Shows how data flows through the system:

- **Data Sources** (CSV, REST API, Database)
- **Window Management** (WindowProvider)
- **Feature Extraction** (FeatureService, FeatureExtractor)
- **Context Management** (JobContext)

### 3. Strategy Pattern View

**File:** [3-strategy-pattern.mmd](3-strategy-pattern.mmd)

Focuses on the analysis strategies:

- **Strategy Interface** (IAnalysisStrategy)
- **Two Implementations:**
  - HeuristicStrategy (Rule-based)
  - SupervisedMLStrategy (ML-based)

### 4. Supervised ML View

**File:** [4-supervised-ml.mmd](4-supervised-ml.mmd)

Details the machine learning pipeline:

- **Feature Engineering** (FeatureExtractor, FeatureVector)
- **ML Detection** (MLDetector)
- **Supervised Learning Strategy**

### 5. Heuristic Components View

**File:** [5-heuristic-components.mmd](5-heuristic-components.mmd)

Details the rule-based diagnosis:

- **Decision Trees** (StrategyTree, PropertyNode)
- **Rule Evaluation**
- **Heuristic Strategy**

## Rendering

These diagrams use Mermaid syntax. You can render them using:

- VS Code with Mermaid Preview extension
- GitHub (renders automatically in markdown)
- Mermaid Live Editor (https://mermaid.live)
- Any documentation system that supports Mermaid
