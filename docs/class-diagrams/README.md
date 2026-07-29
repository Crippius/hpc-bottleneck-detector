# Class Diagram Views

This directory contains multiple views of the HPC Bottleneck Detector architecture, organized by concern for easier understanding.

## Overview

- **[0-full-view.mmd](0-full-view.mmd)** - Complete detailed class diagram with all components and relationships
- **[1-overview.mmd](1-overview.mmd)** - High-level system architecture showing main components and data flow

## Detailed Views

### 2. Data Layer View

**File:** [2-data-layer.mmd](2-data-layer.mmd)

Shows how data flows through the system:

- **Data Sources** (CSVDataSource, XBATDataSource)
- **Window Management** (`DataManager.iterate_windows` / `slice_window`)
- **Context Management** (JobContext, HardwareProfileLoader)

### 3. Supervised ML View

**File:** [3-supervised-ml.mmd](3-supervised-ml.mmd)

Details the machine learning pipeline:

- **Inference Backends** (IMLBackend, DefaultBackend, AMLLibraryBackend) - hold fitted state, used at analysis time
- **Offline Trainers** (IMLTrainer, DefaultTrainer, AMLLibraryTrainer) - produce a fitted backend from labelled CSVs
- **Supervised Learning Strategy** (SupervisedMLStrategy)

### 4. Heuristic Components View

**File:** [4-heuristic-components.mmd](4-heuristic-components.mmd)

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
