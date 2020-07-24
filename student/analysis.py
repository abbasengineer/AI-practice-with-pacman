"""
Analysis question.
Change these default values to obtain the specified policies through value iteration.
If any question is not possible, return just the constant NOT_POSSIBLE:
```
return NOT_POSSIBLE
```
"""

NOT_POSSIBLE = None

def question2():
    """
    By making the Noise = 0, the agent can confidently move EAST
    """

    answerDiscount = 0.9
    answerNoise = 0

    return answerDiscount, answerNoise

def question3a():
    """
    By making a living penalty and 0 noise, the agent confidently hurries to the nearest terminal.
    """

    answerDiscount = 0.9
    answerNoise = 0
    answerLivingReward = -4

    return answerDiscount, answerNoise, answerLivingReward

def question3b():
    """
    Increasing the discount makes the further exit not worth it.
    Increasing the noise makes the agent scared to risk the cliff.
    """

    answerDiscount = 0.2
    answerNoise = 0.2
    answerLivingReward = 0.0

    return answerDiscount, answerNoise, answerLivingReward

def question3c():
    """
    No noise makes the agent confident.
    No living reward allows it to choose an arbitrarily long path
    """

    answerDiscount = 0.9
    answerNoise = 0
    answerLivingReward = 0.0

    return answerDiscount, answerNoise, answerLivingReward

def question3d():
    """
    From the previous question, I increased the noise, making it more risky to take the cliffs
    """

    answerDiscount = 0.9
    answerNoise = 0.2
    answerLivingReward = 0.0
    return answerDiscount, answerNoise, answerLivingReward

def question3e():
    """
    By making everything 0, the agent has no motivation but to do the first safe action (NORTH)
    """

    answerDiscount = 0
    answerNoise = 0
    answerLivingReward = 0
    return answerDiscount, answerNoise, answerLivingReward

def question6():
    """
    I did trial and error, and realized that 50 episodes is too small to learn the optimal policy
    """

    answerEpsilon = 0.3
    answerLearningRate = 0.5
    return NOT_POSSIBLE
    return answerEpsilon, answerLearningRate

if __name__ == '__main__':
    questions = [
        question2,
        question3a,
        question3b,
        question3c,
        question3d,
        question3e,
        question6,
    ]

    print('Answers to analysis questions:')
    for question in questions:
        response = question()
        print('    Question %-10s:\t%s' % (question.__name__, str(response)))
