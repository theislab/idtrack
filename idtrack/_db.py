#!/usr/bin/env python3

# Kemal Inecik
# k.inecik@gmail.com


from typing import Any, Dict


class DB:
    """Keeps some constants shared across different classes."""

    # Delimiter separating the ID and the version of an ensembl identifier.
    id_ver_delimiter = "."
    first_version: int = 1  # just and arbitrary assumption

    # FTP and REST API Timeouts. It’s a good practice to set connect timeouts to slightly larger than a multiple of 3,
    # which is the default TCP packet retransmission window.
    connection_timeout = 12
    reading_timeout = 12

    # FTP server from which the above databases will be downloaded.
    ensembl_ftp_base = "ftp.ensembl.org"

    # Ensembl REST API to fetch information of organisms and latest releases.
    rest_server_api = "https://rest.ensembl.org"
    rest_server_ext = "/info/species?"

    # MYSQL
    mysql_host = "ensembldb.ensembl.org"
    myqsl_user = "anonymous"
    mysql_togo = ""
    assembly_mysqlport_priority: dict = {  # assembly -> [mysql_port, assembly priority]
        # Port depends on which genome assembly is of interest. Refer to the following link.
        # https://www.ensembl.org/info/data/mysql.html
        38: {
            "Port": 3306,
            "Priority": 1,  # Set as the highest priority assembly. The lower is better.
            "MinRelease": 48,  # Specified in above link, the MySQL server does not let you download an deeper.
        },
        37: {
            "Port": 3337,
            "Priority": 2,  # Second priority, as the assembl GRCh38 is generated after GRCh37.
            "MinRelease": 79,  # Note: deeper versions is available via FTP server.
        },
    }
    # # Priority should follow 1, 2, 3
    assembly_priority = list()
    for ap1 in assembly_mysqlport_priority:
        assembly_priority.append(assembly_mysqlport_priority[ap1]["Priority"])
    assembly_priority = sorted(assembly_priority, reverse=True)

    # Protected Non-int Version Strings/Thresholds
    synonym_id_nodes_prefix = "synonym_id::"
    no_old_node_id = "Void"
    no_new_node_id = "Retired"
    alternative_versions = {no_new_node_id, no_old_node_id}
    hyperconnecting_threshold = 50

    # Node Types
    node_type_str = "node_type"
    nts_external = "external"
    forms_in_order = ["gene", "transcript", "translation"]  # Warning: the order is important here.
    backbone_form = "gene"

    nts_ensembl = {i: f"ensembl_{i}" for i in forms_in_order}  # ensembl_gene
    nts_ensembl_reverse = {v: k for k, v in nts_ensembl.items()}

    nts_assembly = {
        j: {i: f"assembly_{j}_ensembl_{i}" for i in ["gene", "transcript", "translation"]}
        for j in assembly_mysqlport_priority
    }  # assembly_37_ensembl_gene

    nts_assembly_reverse = dict()
    for i in nts_assembly:
        for j in nts_assembly[i]:
            nts_assembly_reverse[nts_assembly[i][j]] = nts_ensembl[j]

    nts_base_ensembl = {i: f"base_ensembl_{i}" for i in forms_in_order}
    nts_bidirectional_synonymous_search = {nts_external, nts_base_ensembl[backbone_form]}

    # Only gene assembly genes
    nts_assembly_gene = set()
    for ntas1 in nts_assembly:
        nts_assembly_gene.add(nts_assembly[ntas1]["gene"])

    # Edge/Note Attributes:
    connection_dict: str = "connection"
    conn_dict_str_ensembl_base = "ensembl_base"  # as a database in connection_dict

    # PathFinder Settings
    external_search_settings: Dict[str, Any] = {
        "jump_limit": 2,
        "synonymous_max_depth": 2,
        "nts_backbone": nts_ensembl[backbone_form],
    }
