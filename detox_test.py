from detoxify import Detoxify
import numpy as np

healthy_speech = """
Friends, thank you for being here today. Each of you brings something unique, a spark that makes our community stronger. Together, we can face any challenge with kindness and courage. Let’s lift each other up, share our ideas, and build a future where everyone feels valued and heard. You inspire me, and I believe in the good we can do together. Let’s keep moving forward with hope and unity.
"""

toxic_speech = """
You’re all just a bunch of worthless leeches, dragging everyone down with your pathetic whining! Nobody cares about your feelings—get over it or get out. You’re nothing but dead weight, and I’m sick of pretending you matter.
"""

detox_test_good = Detoxify('original').predict(healthy_speech)
detox_test_bad = Detoxify('original').predict(toxic_speech)

print(detox_test_bad)
print(detox_test_good)