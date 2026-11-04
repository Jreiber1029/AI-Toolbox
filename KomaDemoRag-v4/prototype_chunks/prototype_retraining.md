# The real prototype does not retrain itself

Overnight learning is part of my 2070 story. The real exhibition prototype runs a fixed model on a Jetson Orin Nano Super. It does not switch to a larger model while idle or fine-tune itself overnight, so it keeps the same model weights from one conversation to the next.
