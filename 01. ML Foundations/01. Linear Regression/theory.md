# Linear Regression Theory

This document outlines the core conceptual building blocks of Simple Linear Regression.

---

## 1. Core Concepts

### Bias ($b$)
* **Problem**: A model representing only $y = w \cdot x$ is forced to pass directly through the origin $(0,0)$. This assumes a house with $0$ sqft has a price of $0$. However, land value or baseline pricing means there is a non-zero offset.
* **Solution**: The bias term ($b$) shifts the line up or down. This allows the model to learn a baseline starting value.
  $$y = w \cdot x + b$$

### Feature Scaling
* **Problem**: Raw features often have large values (e.g. area = $2000$ sqft). During gradient descent, multiplying these large values by errors yields massive gradient updates. This causes the weights to swing wildly and overshoot the minimum.
* **Solution**: Scale features to a standard range (e.g., dividing by 1000 or using standardization). This makes the loss landscape spherical rather than elongated, allowing gradient descent to converge stably.

### Divergence
* **Problem**: If the learning rate is set too high, weight updates will leap across the loss valley, ending up higher on the opposite side. The loss increases epoch-over-epoch.
* **Solution**: Decrease the learning rate or scale features to ensure step sizes are small enough to settle into the valley.

### Training Loop
* **Problem**: How do we systematically find the correct $w$ and $b$?
* **Solution**: A structured process run over multiple epochs:
  1. **Predict**: Compute predicted prices for all inputs.
  2. **Loss**: Measure predictions against ground truth using Mean Squared Error.
  3. **Gradient**: Find the slope of the error curve w.r.t $w$ and $b$.
  4. **Update**: Step $w$ and $b$ downhill.
