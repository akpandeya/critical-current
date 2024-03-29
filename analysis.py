from fileinput import filename
import pdb
from typing import List
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

file = "030_DevB1_DC_IV_BaseT_I_12-13_V_21-15_[20_to_-20mT]_[0deg]_OOP_Y-axis_70uT_steps_afterZ-sweep(sameside_config).dat"
exclude = [i for i, line in enumerate(open(file)) if line.startswith("M")]
df = pd.read_csv(file, sep="\t", skiprows=exclude[1:])

mag_field = "Magnetic Field Vector magnitude (T)"
angle = "Vector Angle (q)"
current = "Current (A)"
voltage = "Voltage (V)"

gradient_threshold = 0.5
output_columns = [
    "Magnetic Field (T)",
    "Vector Angle (q)",
    "Ic+(A)",
    "Ic-(A) ",
]
mag_rounding_digits = 3
angle_rounding_digits = 3


def get_sign(x):
    if x > 0:
        return 1
    if x == 0:
        return 0
    return -1


def get_sign_of_series(series: List):
    return [get_sign(x) for x in series]


# Group by for different magnetic field, angle
#  and sign of gradient of current
grouped_data = df.groupby(
    [
        round(df[mag_field], mag_rounding_digits),
        round(df[angle], angle_rounding_digits),
        get_sign_of_series(np.gradient(df[current])),
    ]
)
# print("Keys: ", grouped_data.groups.keys())
# print("Len: ", len(grouped_data.groups.keys()))
# print(
#     "Length of each key",
#     [len(group) for _, group in grouped_data],
# )
output = pd.DataFrame(columns=output_columns)
for i, (key, data) in enumerate(grouped_data):
    fig, ax = plt.subplots()
    ax.plot(data[current], data[voltage], ".")
    ax.set_xlabel(current)
    ax.set_ylabel(voltage)
    ax2 = ax.twinx()
    ax2.plot(
        data[current],
        np.gradient(data[voltage], data[current]),
        ".",
        color="tab:red",
    )
    ax2.set_ylabel(f"dy/dx ({voltage}/{current}")
    max_grad_value = np.max(np.gradient(data[voltage], data[current]))
    max_current = data[
        np.gradient(data[voltage], data[current]) == max_grad_value
    ][current]
    current_label = f"Ic{'+' if key[2] > 0 else '-'}"
    if len(max_current) > 1:
        max_current = np.average(max_current)
    ax2.text(
        max_current,
        max_grad_value,
        f"{current_label} = {float(max_current)}",
    )
    ax2.plot(max_current, max_grad_value, "*", color="orange")

    plt.savefig(f"output/mag={key[0]} angle={key[1]} {current_label}.pdf")
    plt.close()
    ic = np.average(
        data[np.gradient(data[voltage], data[current]) > 0.3][current]
    )
    icp = float(max_current) if key[2] > 0 else None
    icn = float(max_current) if key[2] < 0 else None
    mask = (output[output_columns[0]] == key[0]) & (
        output[output_columns[1]] == key[1]
    )
    if ic:
        if len(output[mask]) == 0:
            output = pd.concat(
                [
                    output,
                    pd.DataFrame(
                        columns=output_columns,
                        data={
                            output_columns[0]: [key[0]],
                            output_columns[1]: [key[1]],
                            output_columns[2]: [icp],
                            output_columns[3]: [icn],
                        },
                    ),
                ]
            )
        else:
            if icp:
                output.loc[mask, output_columns[2]] = icp
            elif icn:
                output.loc[mask, output_columns[3]] = icn
print(output)
output.to_csv(f"output/{file[0:3]}_Summary.csv", index=False)
