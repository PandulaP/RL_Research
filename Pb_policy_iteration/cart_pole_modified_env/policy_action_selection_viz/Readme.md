### How does the (learned) policy select actions?

In this section, I try to design a visualization that showcases what sort of action selections that the learned policy performs during the evaluation phrase.
- For this, I will plot the selected actions against the pendulum angle; here, we try to understand if the action selections align with the expected behaviour (e.g., when the pendulum is angled towards the left-> policy should push the card to left in order to balance it)
- An experimental episode starts from a neutral state and I let the policy performs actions from that state in order to create this visualization.