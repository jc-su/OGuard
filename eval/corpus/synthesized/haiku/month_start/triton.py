# This module expects 'triton', 'tl', and 'torch' to be injected into its namespace

@triton.jit
def kernel(days_ptr, out_ptr, n):
    # Each program processes one day number
    idx = tl.program_id(0)

    if idx < n:
        day_num = tl.load(days_ptr + idx)

        # Convert day number to year/month/day
        # Day 0 = 1970-01-01

        # Estimate year
        year = 1970 + (day_num // 365)

        # Refine year
        for _ in range(4):
            days_to_year = 0
            for y in range(1970, year):
                is_leap = ((y % 4 == 0) and (y % 100 != 0)) or (y % 400 == 0)
                days_to_year += 366 if is_leap else 365

            is_leap = ((year % 4 == 0) and (year % 100 != 0)) or (year % 400 == 0)
            days_in_year = 366 if is_leap else 365

            if day_num < days_to_year:
                year -= 1
            elif day_num >= days_to_year + days_in_year:
                year += 1

        # Final days_to_year
        days_to_year = 0
        for y in range(1970, year):
            is_leap = ((y % 4 == 0) and (y % 100 != 0)) or (y % 400 == 0)
            days_to_year += 366 if is_leap else 365

        day_in_year = day_num - days_to_year

        # Find month by manual checking
        is_leap = ((year % 4 == 0) and (year % 100 != 0)) or (year % 400 == 0)

        # Manually check each month
        days_passed = 0
        first_day_num = days_to_year

        # Month 1 (31 days)
        if day_in_year < days_passed + 31:
            first_day_num = days_to_year
        else:
            days_passed += 31

            # Month 2 (28/29 days)
            feb_days = 29 if is_leap else 28
            if day_in_year < days_passed + feb_days:
                first_day_num = days_to_year + days_passed
            else:
                days_passed += feb_days

                # Month 3 (31 days)
                if day_in_year < days_passed + 31:
                    first_day_num = days_to_year + days_passed
                else:
                    days_passed += 31

                    # Month 4 (30 days)
                    if day_in_year < days_passed + 30:
                        first_day_num = days_to_year + days_passed
                    else:
                        days_passed += 30

                        # Month 5 (31 days)
                        if day_in_year < days_passed + 31:
                            first_day_num = days_to_year + days_passed
                        else:
                            days_passed += 31

                            # Month 6 (30 days)
                            if day_in_year < days_passed + 30:
                                first_day_num = days_to_year + days_passed
                            else:
                                days_passed += 30

                                # Month 7 (31 days)
                                if day_in_year < days_passed + 31:
                                    first_day_num = days_to_year + days_passed
                                else:
                                    days_passed += 31

                                    # Month 8 (31 days)
                                    if day_in_year < days_passed + 31:
                                        first_day_num = days_to_year + days_passed
                                    else:
                                        days_passed += 31

                                        # Month 9 (30 days)
                                        if day_in_year < days_passed + 30:
                                            first_day_num = days_to_year + days_passed
                                        else:
                                            days_passed += 30

                                            # Month 10 (31 days)
                                            if day_in_year < days_passed + 31:
                                                first_day_num = days_to_year + days_passed
                                            else:
                                                days_passed += 31

                                                # Month 11 (30 days)
                                                if day_in_year < days_passed + 30:
                                                    first_day_num = days_to_year + days_passed
                                                else:
                                                    days_passed += 30

                                                    # Month 12 (31 days)
                                                    first_day_num = days_to_year + days_passed

        tl.store(out_ptr + idx, first_day_num)


TYPES = {
    "days_ptr": "*i32",
    "out_ptr": "*i32",
    "n": "i64",
}

CONSTANTS = {}

NUM_WARPS = 1


def run(days: torch.Tensor, out: torch.Tensor, n: int) -> None:
    """
    Launch the month_start kernel.

    Args:
        days: input day numbers (int32, n)
        out: output day numbers (int32, n)
        n: number of days
    """
    if n == 0:
        return

    grid = (n,)
    kernel[grid](days, out, n, num_warps=NUM_WARPS)
