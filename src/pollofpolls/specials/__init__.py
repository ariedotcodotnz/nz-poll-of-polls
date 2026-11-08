"""Special votes in close electorates: from the preliminary count to the official result.

On election night the Electoral Commission counts the ordinary votes (advance and election day). Three to four
weeks later the official count adds the special votes (overseas, out-of-electorate, unpublished roll, not on the
printed roll) and rechecks every ordinary vote. In a close electorate either can change the winner.

The package forecasts the official result of chosen electorates from their preliminary count:

    final = preliminary + change to the ordinary votes in the official count + special votes

``data``      parses the Electoral Commission's official results by voting place (CSV), preliminary results by
              voting place (xlsx) and archived election-night counts
``model``     the hierarchical Bayesian models (NumPyro): special-vote volume, special-vote composition and the
              official-count change to ordinary votes
``simulate``  posterior predictive simulation of the official count
``backtest``  leave-one-election-out evaluation against simple rules of thumb
``run``       the ``pollofpolls specials`` stage: fit, backtest, simulate and write output/specials/

Method: docs/specials.md and the website's "Special votes" page.
"""
