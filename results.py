# -*- coding: utf-8 -*-
"""
Created on Sat Jun 13 14:11:09 2026

@author: Admin
"""


import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# Steady-state result
# ============================================================

class SteadyResult:

    def __init__(
        self,
        network,
        T,
        fluxes,
        G=None,
        convergence=None
    ):

        self.network = network

        self.T = T

        self.fluxes = fluxes

        self.G = G

        self.convergence = convergence

    # --------------------------------------------------------

    def summary(self):

        print("\n")
        print("=" * 60)
        print("CALORIS STEADY RESULTS")
        print("=" * 60)

        print("\nTemperatures\n")

        for node, T in zip(
            self.network.nodes,
            self.T
        ):

            print(
                f"{node.label:<20}"
                f"{T:>12.3f} K"
            )

        print("\nFluxes\n")

        for key, value in self.fluxes.items():

            node_i, node_j, link_type = key

            print(
                f"{node_i} -> {node_j}"
                f" ({link_type})"
                f" : {value:.6f} W"
            )

        if self.convergence is not None:

            print("\nConvergence\n")

            print(
                f"Iterations : "
                f"{len(self.convergence)}"
            )

            print(
                f"Final error : "
                f"{self.convergence[-1]:.3e}"
            )

        print("\n")

    # --------------------------------------------------------

    def export_csv(self, filename):

        df = pd.DataFrame({

            "Node":

            [
                node.label
                for node in self.network.nodes
            ],

            "Temperature [K]":

            self.T

        })

        df.to_csv(
            filename,
            index=False
        )

        print(
            f"Results exported to {filename}"
        )

    # --------------------------------------------------------

    def export_excel(self, filename):

        df_T = pd.DataFrame({

            "Node":

            [
                node.label
                for node in self.network.nodes
            ],

            "Temperature [K]":

            self.T

        })

        flux_rows = []

        for key, value in self.fluxes.items():

            flux_rows.append({

                "From": key[0],

                "To": key[1],

                "Type": key[2],

                "Flux [W]": value

            })

        df_flux = pd.DataFrame(flux_rows)

        with pd.ExcelWriter(filename) as writer:

            df_T.to_excel(
                writer,
                sheet_name="Temperatures",
                index=False
            )

            df_flux.to_excel(
                writer,
                sheet_name="Fluxes",
                index=False
            )

        print(
            f"Results exported to {filename}"
        )


# ============================================================
# Transient result
# ============================================================

class TransientResult:

    def __init__(
        self,
        network,
        time,
        T_history
    ):

        self.network = network

        self.time = time

        self.T_history = T_history

    # --------------------------------------------------------

    def plot(self):

        plt.figure()

        for i, node in enumerate(
            self.network.nodes
        ):

            plt.plot(

                self.time,

                self.T_history[:, i],

                label=node.label

            )

        plt.xlabel("Time [s]")

        plt.ylabel("Temperature [K]")

        plt.title(
            "Transient thermal response"
        )

        plt.grid(True)

        plt.legend()

        plt.show()

    # --------------------------------------------------------

    def plot_node(self, label):

        idx = None

        for i, node in enumerate(
            self.network.nodes
        ):

            if node.label == label:

                idx = i

                break

        if idx is None:

            raise ValueError(
                f"Node {label} not found."
            )

        plt.figure()

        plt.plot(

            self.time,

            self.T_history[:, idx]

        )

        plt.xlabel("Time [s]")

        plt.ylabel("Temperature [K]")

        plt.title(label)

        plt.grid(True)

        plt.show()

    # --------------------------------------------------------

    def export_csv(self, filename):

        data = {

            "Time [s]": self.time

        }

        for i, node in enumerate(
            self.network.nodes
        ):

            data[node.label] = (
                self.T_history[:, i]
            )

        df = pd.DataFrame(data)

        df.to_csv(
            filename,
            index=False
        )

        print(
            f"Transient exported to {filename}"
        )