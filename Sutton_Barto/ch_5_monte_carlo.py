
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

    # Initialize dealer's sum
    dealer_sum = card_value(dealer_card1) + card_value(dealer_card2)
    usable_ace_dealer = 1 in (dealer_card1, dealer_card2)

    # If the dealer sum > 21, he must hold two aces
    if dealer_sum > 21:
        assert dealer_sum == 22
        # Use 1 ace as 1 instead of 11
        dealer_sum -= 10
    
    assert dealer_sum <= 21
    assert player_sum <= 21
    
    # game starts !!!
    
    # players turn
    while True:
        if inital_aciton is not None:
            action = inital_aciton
            inital_aciton = None
        
        else:
            action = policy_player(usable_ace_player, player_sum, dealer_card1)

        # track players trajectory for importance sampling
        player_trajectory.append([usable_ace_player, player_sum, dealer_card1])

        if action == action_stand:
            break
        
        # if hit, get a new card
        card = get_card()

        # Keep track of the ace count. The usable_ace_player alone is insufficient alone as it cannot between having one ace or two
        ace_count = int(usable_ace_player)
        if card == 1:
            ace_count += 1
        
        player_sum += card_value(card)

        # If the player has a usable ace, use it as 1 to avoid busting
        while player_sum > 21 and ace_count:
            player_sum -= 10
            ace_count -= 1
        
        # player bust
        if player_sum > 21:
            return state, -1, player_trajectory
        
        assert player_sum <= 21
        usable_ace_player = (ace_count == 1) # As you can never use more than 1 ace as 11
    
    # Dealer's turn
    while True:

        # get action based on current sum
        action = policy_dealer[dealer_sum]

        if action == action_stand:
            break

        # if hit, get a new card
        new_card = get_card()

        # Keep track of the ace count. The usable_ace_player alone is insufficient alone as it cannot between having one ace or two
        ace_count = int(usable_ace_dealer)
        if new_card == 1:
            ace_count += 1
        
        dealer_sum += card_value(new_card)

        # If the dealer has a usable ace, use it as 1 to avoid busting
        if dealer_sum > 21 and ace_count:
            dealer_sum -= 10
            ace_count -= 1
        
        # Dealer bust
        if dealer_sum > 21:
            return state, 1, player_trajectory
        
        usable_ace_dealer = (ace_count == 1)
    
    # Compare the sum between player and dealer
    assert player_sum <= 21 and dealer_sum <= 21

    if player_sum > dealer_sum:
        return state, 1, player_trajectory
    
    elif player_sum < dealer_sum:
        return state, -1, player_trajectory
    
    else:
        return state, 0, player_trajectory
        
        
def monte_carlo_on_policy(episodes):

    states_usuable_aces = np.zeros((10, 10))

    # Initialize counts to 1 to avoid 0 being divided
    states_usable_ace_count = np.ones((10, 10))

    states_no_usable_ace = np.zeros((10, 10))

    # Initialize counts to 1 to avoid 0 being divided
    states_no_usable_ace_count = np.ones((10, 10))

    for i in range(episodes):
        _, reward, player_trajectory = play(target_policy_play)

        for (usable_ace, player_sum, dealer_card), _ in player_trajectory:
            player_sum -= 12 # no of states
            dealer_card -= 1

            if usable_ace:
                states_usable_ace_count[player_sum, dealer_card] += 1
                states_usuable_aces[player_sum, dealer_card] += reward
            
            else:
                states_no_usable_ace_count[player_sum, dealer_card] += 1
                states_no_usable_ace[player_sum, dealer_card] += reward
    
    return states_usuable_aces/states_usable_ace_count, states_no_usable_ace/states_no_usable_ace_count


def figure_5_1():

    states_usable_ace_1, states_no_usable_ace_1 = monte_carlo_on_policy(10000)
    states_usable_ace_2, states_no_usable_ace_2 = monte_carlo_on_policy(500000)

    states = [states_usable_ace_1, states_usable_ace_2,
              states_no_usable_ace_1, states_no_usable_ace_2]

    titles = ['Usable Ace, 10000 Episodes',
              'Usable Ace, 500000 Episodes',
              'No Usable Ace, 10000 Episodes',
              'No Usable Ace, 500000 Episodes']
    
    _, axes = plt.subplot(2, 2, figsize=(40, 30))
    plt.subplots_adjust(wspace=0.1, hspace=0.2)
    axes = axes.flatten()

    for states, title, axis in zip(states, titles, axes):
        fig = sns.heatmap(np.flipud(states),  cmap="YlGnBu", ax=axis, xticklabels=range(1, 11),
                          yticklabels=list(reversed(range(12, 22)))))
        
        fig.set_ylabel('player sum', fontsize=30)
        fig.set_xlabel('dealer showing', fontsize=30)
        fig.set_title(title, fontsize=30)

    plt.show()
    # plt.savefig('../images/figure_5_1.png')
    # plt.close()

    
if __name__ == "__main__":
    figure_5_1()
    


