""" Water the animal actually CONSUMED this session, for the Stage 1/2 session water cap.

A delivered drop counts as consumed once the animal licks after it was delivered. Until then it
sits at the spout as "pending": a lick in a LATER trial (e.g. during the next WaitQuiescence) still
consumes it, and one lick consumes every pending drop at once (they pool on the spout). Lick times
are each trial's own Bpod-relative Port1In times, the same clock as the Reward state's start.
"""


class ConsumedWaterTracker(object):

    def __init__(self, reward_ul):
        self.reward_ul = float(reward_ul)
        self.delivered_ul = 0.0
        self.consumed_ul = 0.0
        self.pending_ul = 0.0

    def record_trial(self, lick_times, reward_start=None):
        """ lick_times: this trial's Port1In times (trial-relative s). reward_start: the Reward
        state's start on the same clock, or None for an unrewarded trial. """
        lick_times = list(lick_times or [])
        if reward_start is None:
            if lick_times:
                self._consume()
            return
        if any(t < reward_start for t in lick_times):
            self._consume()
        self.delivered_ul += self.reward_ul
        self.pending_ul += self.reward_ul
        if any(t >= reward_start for t in lick_times):
            self._consume()

    def _consume(self):
        self.consumed_ul += self.pending_ul
        self.pending_ul = 0.0
