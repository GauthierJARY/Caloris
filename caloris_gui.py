# -*- coding: utf-8 -*-
"""
Created on Fri Jun 12 22:37:30 2026

@author: Admin
"""
import sys

from PyQt6.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QPushButton,
    QLabel,
    QTableWidget,
    QTableWidgetItem,
    QTextEdit
)


class CalorisWindow(QMainWindow):

    def __init__(self):

        super().__init__()

        self.setWindowTitle("Caloris")
        self.resize(1000, 700)

        central = QWidget()
        self.setCentralWidget(central)

        layout = QVBoxLayout()
        central.setLayout(layout)

        # ==================================================
        # Nodes
        # ==================================================

        layout.addWidget(QLabel("Nodes"))

        self.node_table = QTableWidget()

        self.node_table.setColumnCount(7)

        self.node_table.setHorizontalHeaderLabels([
            "Label",
            "T0 [K]",
            "Q [W]",
            "Mass [kg]",
            "Cp",
            "Boundary Type",
            "Boundary Value"
        ])

        layout.addWidget(self.node_table)

        add_node_btn = QPushButton("Add Node")
        add_node_btn.clicked.connect(self.add_node)

        layout.addWidget(add_node_btn)

        # ==================================================
        # Connections
        # ==================================================

        layout.addWidget(QLabel("Connections"))

        self.connection_table = QTableWidget()

        self.connection_table.setHorizontalHeaderLabels([
            "Node i",
            "Node j",
            "Type",
            "Param1",
            "Param2",
            "Param3",
            "Param4",
            "Param5"
        ])
        self.connection_table.setHorizontalHeaderLabels([
            "Node i",
            "Node j",
            "Type",
            "Parameter"
        ])

        layout.addWidget(self.connection_table)

        add_connection_btn = QPushButton(
            "Add Connection"
        )

        add_connection_btn.clicked.connect(
            self.add_connection
        )

        layout.addWidget(add_connection_btn)

        # ==================================================
        # Solve
        # ==================================================

        solve_btn = QPushButton("Solve")

        solve_btn.clicked.connect(
            self.solve_network
        )

        layout.addWidget(solve_btn)

        # ==================================================
        # Results
        # ==================================================

        layout.addWidget(QLabel("Results"))

        self.results_box = QTextEdit()

        self.results_box.setReadOnly(True)

        layout.addWidget(self.results_box)

    # ======================================================
    # Add Node
    # ======================================================

    def add_node(self):

        row = self.node_table.rowCount()

        self.node_table.insertRow(row)

    # ======================================================
    # Add Connection
    # ======================================================

    def add_connection(self):

        row = self.connection_table.rowCount()

        self.connection_table.insertRow(row)

    # ======================================================
    # Solve
    # ======================================================

    def solve_network(self):
    
        self.results_box.clear()
    
        from nodes import Node
    
        nodes = []
    
        for row in range(self.node_table.rowCount()):
    
            label_item = self.node_table.item(row, 0)
    
            if label_item is None:
                continue
    
            label = label_item.text().strip()
    
            if label == "":
                continue
    
            # -----------------------------
            # T0
            # -----------------------------
    
            T0_item = self.node_table.item(row, 1)
    
            T0 = (
                float(T0_item.text())
                if T0_item and T0_item.text()
                else 300.0
            )
    
            # -----------------------------
            # Q
            # -----------------------------
    
            Q_item = self.node_table.item(row, 2)
    
            Q = (
                float(Q_item.text())
                if Q_item and Q_item.text()
                else 0.0
            )
    
            # -----------------------------
            # Mass
            # -----------------------------
    
            mass_item = self.node_table.item(row, 3)
    
            mass = (
                float(mass_item.text())
                if mass_item and mass_item.text()
                else 1.0
            )
    
            # -----------------------------
            # Cp
            # -----------------------------
    
            cp_item = self.node_table.item(row, 4)
    
            cp = (
                cp_item.text()
                if cp_item and cp_item.text()
                else None
            )
    
            # -----------------------------
            # Boundary
            # -----------------------------
    
            bc_type_item = self.node_table.item(row, 5)
    
            bc_type = (
                bc_type_item.text()
                if bc_type_item and bc_type_item.text()
                else None
            )
    
            bc_value_item = self.node_table.item(row, 6)
    
            bc_value = (
                float(bc_value_item.text())
                if bc_value_item and bc_value_item.text()
                else None
            )
    
            node = Node(
                label=label,
                initial_temperature=T0,
                constant_heat_input=Q,
                mass=mass,
                specific_heat=cp,
                boundary_type=bc_type,
                boundary_value=bc_value
            )
    
            nodes.append(node)
    
        self.results_box.append(
            f"{len(nodes)} nodes created\n"
        )
    
        for node in nodes:
    
            self.results_box.append(
                str(node)
            )
        
        from connections import Connection
        node_dict = {
            node.label: node
            for node in nodes
        }
        connections = []

        for row in range(
            self.connection_table.rowCount()
        ):
        
            item_i = self.connection_table.item(row,0)
            item_j = self.connection_table.item(row,1)
            item_type = self.connection_table.item(row,2)
        
            if (
                item_i is None
                or item_j is None
                or item_type is None
            ):
                continue
        
            label_i = item_i.text().strip()
            label_j = item_j.text().strip()
        
            type_ = item_type.text().strip()
        
            node_i = node_dict[label_i]
            node_j = node_dict[label_j]
            if type_ == "conduction":
                A = float(
                    self.connection_table.item(row,3).text()
                )
            
                L = float(
                    self.connection_table.item(row,4).text()
                )
            
                k = self.connection_table.item(row,5).text()
            
                try:
                    k = float(k)
                except:
                    pass
            
                conn = Connection(
                    node_i,
                    node_j,
                    connection_type="conduction",
                    A=A,
                    L=L,
                    material_conductivity=k
                )
            elif type_ == "radiation":

                    ei = float(
                        self.connection_table.item(row,3).text()
                    )
                
                    ej = float(
                        self.connection_table.item(row,4).text()
                    )
                
                    Si = float(
                        self.connection_table.item(row,5).text()
                    )
                
                    Sj = float(
                        self.connection_table.item(row,6).text()
                    )
                
                    Fij = float(
                        self.connection_table.item(row,7).text()
                    )
                
                    conn = Connection(
                        node_i,
                        node_j,
                        connection_type="radiation",
                        e_i=ei,
                        e_j=ej,
                        S_i=Si,
                        S_j=Sj,
                        F_ij=Fij
                    )
            elif type_ == "conductance":
        
                G = float(
                    self.connection_table.item(row,3).text()
                )
            
                conn = Connection(
                    node_i,
                    node_j,
                    connection_type="conductance",
                    G=G
                )
            connections.append(conn)
                
if __name__ == "__main__":

    app = QApplication(sys.argv)

    window = CalorisWindow()

    window.show()

    sys.exit(app.exec())