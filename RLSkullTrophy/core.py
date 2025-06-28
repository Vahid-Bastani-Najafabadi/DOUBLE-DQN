import numpy as np

class SkullTrophyRLEnv:
  def __init__(self):
    self.__envAlive = False
    self.__query = '-1'
    self.__envData = {}
    self.__beginEnvironment()
    self.__state = 18
    self.__totalStates = 19
    self.__isEpisodeRunning = False

  
  def __beginEnvironment(self):
    self.__setQuery()
    self.__loadEnvironment()
    self.__envAlive = True
  
  def beginEpisode(self, s0=18):
    if not isinstance(s0, int):
      raise TypeError('starting state must be an integer with min value of 0 and a max value of 18')
    if not (0 <= s0 <= 18):
      raise ValueError('starting state must be an integer with min value of 0 and a max value of 18')
    self.__state = s0
    self.__isEpisodeRunning = True
    return self.__handleNewState()
  
  def terminateEpisode(self):
    self.__isEpisodeRunning = False
    return self.__handleNewState()

  def step(self, action):
    possibleStates = range(self.__totalStates)
    probabilities = self.__envData['T'][self.__state, action, :]
    self.__state = np.random.choice(possibleStates, p=probabilities)

    return self.__handleNewState()

  def __handleNewState(self):
    if self.__envData['isTerminal'][self.__state] == 1:
      self.__isEpisodeRunning = False
    return self.__envData['images'][self.__state], self.__envData['R'][self.__state], self.__isEpisodeRunning

  def __setQuery(self):
    self.__query = '5DLx0ZM-RLlbH0rKd-8tTHZNI7WKHazLS9sXGKK0-Z0='
  
  def __loadEnvironment(self):
    import pickle
    from cryptography.fernet import Fernet
    import importlib.resources as pkg_resources
    from . import data
    with pkg_resources.files(data).joinpath("Environment_Variables.pkl").open("rb") as f:
      enc = f.read()
    cipher = Fernet(self.__query)
    dec = cipher.decrypt(enc)
    self.__envData = pickle.loads(dec)