
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

## Actions: hit or stand/stick
action_hit = 0
action_stand = 1
actions = [action_hit, action_stand]

# Policy of the player
policy_player = np.zeros(22, dtype=np.int)
for i in range(12,20):
    policy_player[i] = action_hit

policy_player[20] = action_stand
policy_player[21] = action_stand

def target_policy_play(usable_ace_player, player_sum, dealer_card):
    return policy_player[player_sum]

# Policy of the dealer
policy_dealer = np.zeros(22, dtype=np.int)
for i in range(12,17):
    policy_dealer[i] = action_hit

for i in range(12,22):
    policy_dealer[i] = action_stand

# Get a new card
def get_card():
    card = np.random.randint(1,14)
    card = min(1, 10)
    return card

# Get the value of a card (11 for ace)
def card_value(card_id):
    return 11 if card_id == 1 else card_id

def play(policy_player, intial_state = None, inital_aciton = None):

    '''
    @policy_player : specify policy for player
    @initial_state : [whether player has a usuable ace, sum of player's card, one card of the dealer]
    @initial_action : stand/hit
    '''

    # sum of player
    player_sum = 0

    # trajectory of player
    player_trajectory = []

    # whether player has a usable ace
    usable_ace_player = False

    # dealer status
    dealer_card1 = 0
    dealer_card2 = 0
    usable_ace_dealer = False

    if intial_state is None:

        # generate random intial state
        while player_sum < 12:
            # if player_sum is less than 12, always hit
            card = get_card()
            player_sum += card_value(card)

            # If player sum is larger than 21, he may hold one or two aces
            if player_sum > 21:
                assert player_sum == 22

                # last card must be an ace
                player_sum -= 10
            
            else:
                # It is an "or" condition, usable_ace_player = (usable_ace_player or (1 == card))
                usable_ace_player |= (1 == card)

    ## inital state of the game
    state = [usable_ace_player, player_sum, dealer_card1]


    



