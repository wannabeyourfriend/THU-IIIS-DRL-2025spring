# Deep Reinforcement Learning (DRL)

**Course:** Deep Reinforcement Learning
**Institution:** Tsinghua University - Institute for Interdisciplinary Information Sciences (IIIS)
**Semester:** Spring 2025

## Overview

This repository contains course materials and assignments for the Deep Reinforcement Learning course. It includes 4 programming assignments and 1 final project covering fundamental concepts and algorithms in reinforcement learning.

## Course Structure

- **4 Programming Assignments** covering:
  - Dynamic Programming (DP)
  - Temporal-Difference (TD) Learning
  - Deep Deterministic Policy Gradient (DDPG)
  - Soft Actor-Critic (SAC)
  - Twin Delayed DDPG (TD3)
- **1 Final Project**

## Repository Structure

```
.
├── Lab/           # Laboratory exercises and implementations
│   ├── DP/        # Dynamic Programming labs
│   └── TD/        # Temporal-Difference Learning labs
├── Note/          # Course notes and supplementary materials
├── PA/            # Programming assignments
│   ├── PA1_dp_td/
│   ├── PA2_*/
│   └── PA3_ddpg_sac_td3/
└── README.md      # This file
```

## Resources

### Laboratory Exercises

- [DP Learning - Cliff Walking](https://github.com/wannabeyourfriend/RL-DP-Cliff-walking)
- [TD Learning - Cliff Walking](https://github.com/wannabeyourfriend/RL-TD-Cliff-walking)

## Prerequisites

- Python 3.8+
- PyTorch
- NumPy
- Gym/Gymnasium
- Jupyter Notebook
- Matplotlib

## Installation

```bash
# Clone the repository
git clone https://github.com/wannabeyourfriend/THU-IIIS-DRL-2025spring.git
cd THU-IIIS-DRL-2025spring

# Install dependencies (if requirements.txt exists)
pip install -r requirements.txt
```

## Usage

Each programming assignment and lab includes its own instructions. Please refer to the specific directories for detailed information.

### Running Labs

```bash
cd Lab/DP  # or Lab/TD
python main.py
```

### Running Assignments

```bash
cd PA/PA1_dp_td  # or other assignment folders
python test_main.py
```

## Academic Integrity Policy

**Tsinghua University Student Discipline Management Regulations**

### Chapter VI: Academic Misconduct and Violation of Study Discipline

Article 21: Students who commit any of the following violations of course study discipline shall receive a penalty ranging from warning to probation:

1. Serious plagiarism in course assignments
2. Serious plagiarism in laboratory reports or falsification of experimental data
3. Serious plagiarism in mid-term or final course papers
4. Other serious acts of falsification during the course of study

**Please ensure all work submitted is your own. Properly cite and reference any external resources used.**

## Contributing

This repository is for course purposes only. For questions or suggestions, please open an issue or contact the course instructor.

## License

This repository is for educational purposes. Please refer to individual assignments for specific usage guidelines.

## Acknowledgments

- Course instructor and teaching assistants
- Tsinghua University IIIS
- Open source RL community

## Contact

For course-related inquiries, please contact the course instructor or teaching assistants through official channels.

---

**Note:** This repository is maintained for educational purposes. Always adhere to your institution's academic integrity policies.
