# Collaborative-Transport-With-Model-Predictive-Control
Decentralized MPC-based formation control for collaborative warehouse transport using four differential-drive robots. Compares single-leader and dual-leader strategies in simulation, analyzing path tracking, formation stability, and performance across straight, curved, and turning trajectories.

# Collaborative Transport Using Warehouse Robots with MPC

This project explores **formation control for collaborative warehouse transport** using a team of four autonomous differential-drive robots. Each robot is modeled with unicycle kinematics and controlled via **decentralized Model Predictive Control (MPC)**. The goal is to maintain a square formation while tracking predefined paths, enabling cooperative transport of larger payloads across warehouse environments.

---

## 📖 Overview
The study compares two coordination strategies:

- **Single-Leader Strategy**  
  One robot acts as the leader, directly tracking the global path. The others follow offset paths while adjusting controls to maintain formation relative to the leader.

- **Dual-Leader Strategy**  
  Two front robots act as independent leaders, each tracking offset paths. The rear robots follow their respective leader, prioritizing formation stability.

Simulations were performed in Python using **CVXPY** and the **Clarabel solver**, testing three representative warehouse maneuvers:
1. Straight-line traversal  
2. Obstacle avoidance (curved path)  
3. 90° turn maneuver  

---

## ⚙️ Methodology
- **Robot Model:** Discrete-time unicycle (position, heading, velocity states; acceleration and angular velocity inputs).  
- **MPC Formulation:**  
  - Predicts robot states over a finite horizon.  
  - Minimizes path-tracking error, formation deviation, and control effort.  
  - Subject to kinematic constraints and dynamic limits.  
- **Enhancements:**  
  - Curvature-aware speed profiles  
  - Formation constraint relaxation  
  - Speed boosting for lagging robots  
  - Path unlocking to prevent overtaking  

---

## 📊 Results
- **Single-Leader Strategy:**  
  - Faster traversal times  
  - Strong performance during sharp turns  
  - Weaker formation stability on straight/curved paths  

- **Dual-Leader Strategy:**  
  - Superior formation maintenance on straight and curved paths  
  - Weaker performance during sharp turns  
  - Higher computational and coordination demand  

- **Key Insight:**  
  A **hybrid strategy** that switches between single-leader and dual-leader modes depending on path curvature would achieve the best overall performance.

---

## 🚀 Future Work
- Implement **dynamic hybrid switching** between strategies.  
- Extend to larger robot teams and real-world warehouse tests.  
- Incorporate perception and obstacle detection into MPC.  
