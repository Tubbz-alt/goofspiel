# goofspiel
An array of algorithms and experiments on playing Goofspiel.

You have to install openspiel for the code to run. It unfortunately takes more than a simple one-liner to do so.
Follow the instructions in this paper for the installation: https://arxiv.org/pdf/1908.09453.pdf

You can use either conda_requirements.txt or conda_requirements.txt depending on your virtual environment preference.

The jupyter script is intended to give a glimpse of how pyspiel (openspiel) works. It simulates 1000 games between a
random agent who picks their next action u.a.r, and a deterministic agent who always bids the face value of the latest
point card. Unfortunately, there is no documentation for pyspiel, so we have to understand how to use it by looking at
the source code. So far, this has been pretty straightforward.
