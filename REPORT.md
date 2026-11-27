# Findings for the support lead

**The result.** At 400 training examples, the best baseline (TF-IDF with logistic regression) scored 0.885 on the validation rows and 0.867 on the test rows. The best fine-tuned method (DistilBERT, whole model) scored 0.919 on validation and 0.883 on test.

**Where fine-tuning helped.** Fine-tuning the whole model beat TF-IDF at all three sizes, and the ranges over three seeds did not overlap: by 0.07 at 50 examples, 0.08 at 150 and 0.03 at 400. The gap shrinks as TF-IDF gets more data. Training only the new head never clearly beat TF-IDF: it tied at 50 and was worse at 150 and 400. On the test rows the full model leads by 0.017, about four messages out of 240, from a single run, so I cannot call it a clear win.

**Why I think so.** First, repeated wording helps the cheap baseline. The generated messages reuse sentence patterns, and TF-IDF rose from 0.64 to 0.89 as it saw more of them. My own messages point the same way (below). Second, some labels cannot be known from the text. I counted 140 vague messages out of 1,200, about 12%, which puts the ceiling near 0.9; this is a calculation, not an experiment. Third, noise from seeds: I ran three seeds and report ranges, and head-only at 400 examples ranged from 0.725 to 0.825. Fourth, the head-only learning rate: I tried two values on one size and one seed and did not check further, so head-only may be under-tuned.

**My own 40 messages.** The full model got 37 of 40 right (0.925) and TF-IDF 32 (0.800), so the result points the same way as the test result and was not worse. TF-IDF scored lower on my wording than on the test rows, which fits the idea that it leans on repeated phrases. But my messages contain no vague or mixed ones, so they are easier, and 40 is too few to be sure the generated data is not flattering the models.

**Recommendation.** Use the fine-tuned model to suggest a queue that a person confirms, because it was ahead at every size and on my own messages. One thing that would change this: if a test on real messages shows the gap disappearing, the cheaper TF-IDF is the better choice.

**Honest limit.** The training text was invented, so my evidence cannot tell you how either method behaves on real customers. I used three seeds and one test set, tuned settings on one size and one seed, and did not tune the baseline.