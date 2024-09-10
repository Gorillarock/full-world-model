obs = env.reset()  # observation initialized to the reset state of the environment
h = mdnrnn.initial_State # initial state of the RNN (hidden state)
done = False
cummulative_reward = 0

while not done:
    # get the input frame
    z = cnnvae(obs)  # returns latent vector z
    a = controller([z, h])  # returns action a, given the concatenated input vector of latent vector z and the hidden state h
    obs, reward, done = env.step(a)  # returns the next observation, reward, done flag
    cummulative_reward += reward # can be a negative reward
    h = mdnrnn([a, z, h])  # update the next hidden state h, given the action a, latent vector z, and the previous hidden state fed back into the mdnrnn
