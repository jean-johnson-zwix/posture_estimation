# Posture Estimation

## Phase 1: Detect Posture using Media Pipe

### Media Pipe

MediaPipe Solutions provides a suite of libraries and tools for you to quickly apply artificial intelligence (AI) and machine learning (ML) techniques in your applications.

#### Pose Landmark Detection

The MediaPipe Pose Landmarker task lets you detect landmarks of human bodies in an image or video. You can use this task to identify key body locations, analyze posture, and categorize movements. This task uses machine learning (ML) models that work with single images or video. The task outputs body pose landmarks in image coordinates and in 3-dimensional world coordinates.

*Model Used:* Pose landmarker (lite)
*Mode*: VIDEO

## Phase 2: Estimate Posture based on Body Angles

### Posture Angle Computation

Compute the following measurements:

- Torso Lean
- Forward Head Bending
- Shoulder Tilt
- Leg Angle

### Posture Estimation

Based on the computed angles, estimate posture

- if leg angle less than 50, then 'standing'
- if torso lean less than 25, then 'poor' posture
- if forward head bending less than 15, then 'poor' posture
- if shoulder tilt less than 10, then 'poor' posture
