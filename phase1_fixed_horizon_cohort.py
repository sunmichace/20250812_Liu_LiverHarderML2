"""Compatibility entry point for the fixed-horizon cohort build.

The visit cleaning and cohort construction are deliberately inseparable: target
selection must occur only after same-day LSM aggregation and time-safe clinical
value matching.  This wrapper keeps the planned phase name while calling phase 0.
"""

from phase0_final_data_preparation import main


if __name__ == "__main__":
    main()

