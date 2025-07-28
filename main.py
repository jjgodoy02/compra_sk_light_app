# -*- coding: utf-8 -*-
"""
Created on Tue Jul  8 10:22:56 2025

@author: Jose Godoy
"""

# ─────────────────────────────────────────────────────────────────────────────
# 1. Librerías
# ─────────────────────────────────────────────────────────────────────────────
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import io

# ─────────────────────────────────────────────────────────────────────────────
# 2. Funciones de carga de datos
# ─────────────────────────────────────────────────────────────────────────────
def cargar_catalogo():
    """
    Permite al usuario subir el archivo de catálogo en Excel.
    Retorna el dataframe con las columnas necesarias convertidas a texto.
    """

    archivo = st.file_uploader("Sube el archivo del catálogo (.xlsx)", type="xlsx")

    if archivo:
        try:
            df = pd.read_excel(archivo)
            df['STYLE'] = df['STYLE'].astype(str)
            df['COLOR'] = df['COLOR'].astype(str)
            df['SKU'] = df['SKU'].astype(str)
            return df
        except Exception as e:
            st.error(f"Error al cargar el archivo de catálogo: {e}")
            return pd.DataFrame()
    else:
        st.info("Por favor, sube un archivo de catálogo en formato Excel (.xlsx).")
        return pd.DataFrame()

def cargar_ventas():
    """
    Permite al usuario subir manualmente archivos de ventas en formato CSV.
    Extrae el país desde el nombre del archivo y procesa las fechas.
    """

    archivos = st.file_uploader("Sube los archivos de ventas (.csv)", type="csv", accept_multiple_files=True)
    ventas_dfs = []

    if archivos:
        for archivo in archivos:
            try:
                filename = archivo.name
                pais = filename.split('_')[1].split('-')[0]
                pais = "ES" if pais == "SV" else pais

                df_sales = pd.read_csv(archivo)
                df_sales['Pais'] = pais
                df_sales['Fecha'] = pd.to_datetime(
                    df_sales[['Anio', 'Mes', 'Dia']].rename(columns={'Anio': 'year', 'Mes': 'month', 'Dia': 'day'})
                )
                df_sales['DivisionGenero'] = df_sales['U_Division'].astype(str) + " " + df_sales['U_Genero'].astype(str)

                ventas_dfs.append(df_sales)

            except Exception as e:
                st.error(f"Error procesando {filename}: {e}")

        if ventas_dfs:
            df_all = pd.concat(ventas_dfs, ignore_index=True)
            df_all['YearMonth'] = df_all['Fecha'].dt.to_period('M').astype(str)
            ventas_mensuales = (
                df_all.groupby(['YearMonth', 'Pais', 'DivisionGenero'], as_index=False)['Cantidad']
                .sum()
                .rename(columns={'Cantidad': 'Ventas'})
            )
            return ventas_mensuales
    return pd.DataFrame(columns=['YearMonth', 'Pais', 'DivisionGenero', 'Ventas'])

def cargar_presupuesto():
    archivo = st.file_uploader("Sube el archivo de presupuesto (.xlsx)", type="xlsx")
    
    if archivo:
        try:
            df = pd.read_excel(archivo)
            num_cols = df.select_dtypes(include='number').columns
            df[num_cols] = (np.ceil(df[num_cols] / 12) * 12).astype(int)
            return df
        except Exception as e:
            st.error(f"Error al cargar el archivo: {e}")
            return pd.DataFrame()
    else:
        st.info("Por favor, sube un archivo de presupuesto en formato Excel (.xlsx).")
        return pd.DataFrame()

# ─────────────────────────────────────────────────────────────────────────────
# 3. Helper functions
# ─────────────────────────────────────────────────────────────────────────────
# Función para checkear que se haya introducido un texto válido (Número entre 0-10, o vacío)
def validar_entrada(texto):
    if texto == "":
        return True, None
    try:
        val = float(texto)
        if 0 <= val <= 10:
            return True, val
        else:
            return False, None
    except:
        return False, None
    
def greedy_rounder(relative_scores, budget):
    if relative_scores.isna().all():
        return pd.Series(0, index=relative_scores.index)
    
    budget = budget.iloc[0]
    relative_scores = relative_scores.fillna(0)
    raw_alloc = relative_scores * budget
    dozens_raw = raw_alloc / 12
    dozens_floored = np.floor(dozens_raw)

    final_dozens = pd.Series(dozens_floored, index=relative_scores.index)

    budget_dozens = int(budget) // 12
    current_sum = int(final_dozens.sum())
    diff = budget_dozens - current_sum

    frac_parts = dozens_raw - dozens_floored
    indices = np.argsort(-frac_parts.to_numpy())  # posiciones, no etiquetas

    if diff > 0:
        for i in indices[:diff]:
            final_dozens.iloc[i] += 1
    elif diff < 0:
        for i in indices[:abs(diff)]:
            if final_dozens.iloc[i] > 0:
                final_dozens.iloc[i] -= 1

    final_alloc = (final_dozens * 12).astype(int)
    return final_alloc

def random_number(std_dev=2, lower=0, upper=10):
    mean = np.random.uniform(3, 7)
    x = np.random.normal(loc=mean, scale=std_dev)
    x= np.clip(x, lower, upper)
    return x

def random_boolean(prob_1=0.5):  
    return np.random.choice([0, 1], p=[1 - prob_1, prob_1])


# ─────────────────────────────────────────────────────────────────────────────
# 4. Configuración inicial
# ─────────────────────────────────────────────────────────────────────────────
cmap = plt.cm.magma
new_cmap = mcolors.LinearSegmentedColormap.from_list('magma_trunc', cmap(np.linspace(0, 0.3, 256)))

if not st.session_state.get("archivos_subidos", False):
    st.set_page_config(page_title="Bienvenido", page_icon="📦", layout="wide")

    st.title("📦 Bienvenido al Calificador de Productos")
    st.markdown("Por favor, sube los archivos necesarios para comenzar:")

    # File uploaders
    catalogo_file = st.file_uploader("🗂 Archivo de preventa (.xlsx)", type="xlsx", key="catalogo")
    presupuesto_file = st.file_uploader("💰 Presupuesto (.xlsx)", type="xlsx", key="presupuesto")
    ventas_files = st.file_uploader("📊 Ventas históricas (.csv)", type="csv", accept_multiple_files=True, key="ventas")

    # Botón para continuar cuando todos los archivos están subidos
    if catalogo_file and presupuesto_file and ventas_files:
        if st.button("🚀 Comenzar a calificar"):
            st.session_state["catalogo_file"] = catalogo_file
            st.session_state["presupuesto_file"] = presupuesto_file
            st.session_state["ventas_files"] = ventas_files
            st.session_state["archivos_subidos"] = True
            st.rerun()
    else:
        st.warning("Sube todos los archivos requeridos para continuar.")

if st.session_state.get("archivos_subidos", False):
    # Cargar catálogo
    if 'df' not in st.session_state:
        df = pd.read_excel(st.session_state["catalogo_file"])
        df['STYLE'] = df['STYLE'].astype(str)
        df['COLOR'] = df['COLOR'].astype(str)
        df['SKU'] = df['SKU'].astype(str)

        # Asegurar columnas de países
        new_cols = ['GT', 'ES', 'RD', 'HN TGU', 'HN SPS']
        for col in new_cols:
            if col not in df.columns:
                df[col] = 0

        # Asegurar columnas de score
        new_score_cols = ['GT score', 'ES score', 'RD score', 'HN TGU score', 'HN SPS score']
        for col in new_score_cols:
            if col not in df.columns:
                df[col] = None

        st.session_state.df = df.copy()
    else:
        df = st.session_state.df

    # Cargar ventas históricas
    if 'ventas_historicas' not in st.session_state:
        ventas_dfs = []
        for archivo in st.session_state["ventas_files"]:
            try:
                filename = archivo.name
                pais = filename.split('_')[1].split('-')[0]
                pais = "ES" if pais == "SV" else pais

                df_sales = pd.read_csv(archivo)
                df_sales['Pais'] = pais
                df_sales['Fecha'] = pd.to_datetime(
                    df_sales[['Anio', 'Mes', 'Dia']].rename(columns={'Anio': 'year', 'Mes': 'month', 'Dia': 'day'})
                )
                df_sales['DivisionGenero'] = df_sales['U_Division'].astype(str) + " " + df_sales['U_Genero'].astype(str)

                ventas_dfs.append(df_sales)

            except Exception as e:
                st.error(f"Error procesando {archivo.name}: {e}")

        if ventas_dfs:
            df_all = pd.concat(ventas_dfs, ignore_index=True)
            df_all['YearMonth'] = df_all['Fecha'].dt.to_period('M').astype(str)
            ventas_historicas = (
                df_all.groupby(['YearMonth', 'Pais', 'DivisionGenero'], as_index=False)['Cantidad']
                .sum()
                .rename(columns={'Cantidad': 'Ventas'})
            )
        else:
            ventas_historicas = pd.DataFrame(columns=['YearMonth', 'Pais', 'DivisionGenero', 'Ventas'])

        st.session_state.ventas_historicas = ventas_historicas.copy()
    else:
        ventas_historicas = st.session_state.ventas_historicas

    # Cargar presupuesto
    if 'budget' not in st.session_state:
        try:
            budget = pd.read_excel(st.session_state["presupuesto_file"])
            num_cols = budget.select_dtypes(include='number').columns
            budget[num_cols] = (np.ceil(budget[num_cols] / 12) * 12).astype(int)
            st.session_state.budget = budget.copy()
        except Exception as e:
            st.error(f"Error cargando presupuesto: {e}")
            st.session_state.budget = pd.DataFrame()
    else:
        budget = st.session_state.budget
    
# ─────────────────────────────────────────────────────────────────────────────
# 5. Selección de división o grupo
# ─────────────────────────────────────────────────────────────────────────────
if st.session_state.get("archivos_subidos", False):
    st.set_page_config(layout="wide")

    display_choice = st.sidebar.radio(
        "Seleccione una opción para mostrar los productos:",
        ["Por división", "Por orden de catálogo"])

    st.sidebar.write("")

    if display_choice == "Por división":
        divisiones = sorted(df['DIVISION NAME'].unique())
        div_options =  ['Todas las divisiones'] + divisiones
        division_seleccionada = st.sidebar.selectbox("Selecciona una División", div_options)
        
        if division_seleccionada == "Todas las divisiones":
            df_div = st.session_state.df.copy()
        else:
            df_div = st.session_state.df[st.session_state.df['DIVISION NAME'] == division_seleccionada].copy()
        
        grouped = df_div.groupby('STYLE', sort=False)
        
        st.title(f"PREVENTA SS26 - {division_seleccionada}")

    if display_choice == "Por orden de catálogo":
        groups = df['GrupoCatalogo'].unique()
        group_options =  groups
        grupo_seleccionado = st.sidebar.selectbox("Selecciona un Grupo", group_options)
        st.sidebar.write("")
        
        if grupo_seleccionado == "Todos los grupos":
            df_div = st.session_state.df.copy()
        else:
            df_div = st.session_state.df[st.session_state.df['GrupoCatalogo'] == grupo_seleccionado].copy()
        
        grouped = df_div.groupby('STYLE', sort=False)
        
        st.title(f"PREVENTA SS26 - {grupo_seleccionado}")
        

# ─────────────────────────────────────────────────────────────────────────────
# 6. Presupuesto de la división seleccionada
# ─────────────────────────────────────────────────────────────────────────────
    if display_choice == 'Por división':
        if division_seleccionada == 'Todas las divisiones': 
            budget_div = budget.iloc[:-1].copy()   
            budget_div['Total'] = budget_div.drop(columns=('DIVISION NAME')).sum(axis=1)
            subset_for_gradient = budget_div.columns.difference(['DIVISION NAME'])
            subset_for_format = budget_div.columns.difference(['DIVISION NAME'])
            cmap = plt.cm.magma
            new_cmap = mcolors.LinearSegmentedColormap.from_list(
                'magma_trunc', cmap(np.linspace(0, 0.3, 256))
            )
            gradient_args = {
            'cmap': new_cmap,
            'subset': subset_for_gradient,
            'axis': 0
            }
            
            fmt_dict = {col: "{:.0f}" for col in subset_for_format}
            styled_budget_div = (budget_div.sort_values('DIVISION NAME')
                                 .style
                                 .background_gradient(**gradient_args)
                                 .format(fmt_dict)
            )
            st.markdown("#### Presupuesto por país")
            st.dataframe(styled_budget_div, hide_index=True)
            
        else:
            budget_div = budget[budget['DIVISION NAME'] == division_seleccionada].copy()  
            budget_div['Total'] = budget_div.drop(columns=('DIVISION NAME')).sum(axis=1)
            subset_for_gradient = budget_div.columns.difference(['Total', 'DIVISION NAME'])
            subset_for_format = budget_div.columns.difference(['DIVISION NAME'])
            gradient_args = {
            'cmap': new_cmap,
            'subset': subset_for_gradient,
            'axis': 1
            }
            
            fmt_dict = {col: "{:.0f}" for col in subset_for_format}
            styled_budget_div = (budget_div.sort_values('DIVISION NAME')
                                 .style
                                 .background_gradient(**gradient_args)
                                 .format(fmt_dict)
            )
            st.markdown("#### Presupuesto por país")
            st.dataframe(styled_budget_div, hide_index=True)
            
    if display_choice == 'Por orden de catálogo':
        budget_div = budget.iloc[:-1].copy()   
        budget_div['Total'] = budget_div.drop(columns=('DIVISION NAME')).sum(axis=1)
        subset_for_gradient = budget_div.columns.difference(['DIVISION NAME'])
        subset_for_format = budget_div.columns.difference(['DIVISION NAME'])
        
        cmap = plt.cm.magma
        new_cmap = mcolors.LinearSegmentedColormap.from_list(
            'magma_trunc', cmap(np.linspace(0, 0.3, 256))
        )
        gradient_args = {
        'cmap': new_cmap,
        'subset': subset_for_gradient,
        'axis': 0
        }
        
        fmt_dict = {col: "{:.0f}" for col in subset_for_format}
        styled_budget_div = (budget_div.sort_values('Total',ascending=False)
                             .style
                             .background_gradient(**gradient_args)
                             .format(fmt_dict)
        )
            
        
# ─────────────────────────────────────────────────────────────────────────────
# 7. Estado de la compra de la división seleccionada
# ─────────────────────────────────────────────────────────────────────────────
    if display_choice == 'Por división':
        st.markdown("#### Estado de la compra")
        if division_seleccionada == 'Todas las divisiones':
            score_cols = ['GT score','ES score','HN TGU score','HN SPS score','RD score']
            matrixes = {}
            for division in st.session_state.df['DIVISION NAME'].unique():
                subdf = st.session_state.df[st.session_state.df['DIVISION NAME']==division].copy()
                
                suggested_matrix_of_division = pd.DataFrame(
                0,
                index=subdf.index,
                columns=budget_div.drop(columns='DIVISION NAME').columns
                )
                suggested_matrix_of_division['SKU'] = subdf['SKU']
                budget_specific = budget_div[budget_div['DIVISION NAME']==division]
                
                
                for col in score_cols:
                    country = col.split('score')[0].strip()
                    total_scores = subdf[col].sum()

                    relative_col = f'relative {col}'
                    subdf[col] = pd.to_numeric(subdf[col], errors='coerce')
                    total_scores = pd.to_numeric(total_scores, errors='coerce')
                    subdf.loc[:, relative_col] = (subdf[col] / total_scores)
                    budget = budget_specific[country]
                    
                    suggested_col = greedy_rounder(subdf[relative_col], budget)

                    suggested_matrix_of_division[country] = suggested_col
                    
                matrixes[division] = suggested_matrix_of_division

                # Actualizar valores solo de las columnas relevantes (países)
                st.session_state.df.set_index('SKU').update(suggested_matrix_of_division)

                    
            dfs = [df for df in matrixes.values() if isinstance(df, pd.DataFrame)]
            
            suggested_matrix = pd.concat(dfs)
            
            suggested_matrix['Total'] = suggested_matrix.drop(columns='SKU').sum(axis=1)
            gradient_args = {'cmap': new_cmap, 'subset': subset_for_gradient, 'axis': 0}   
            
            
            totals = pd.DataFrame(columns=suggested_matrix.drop(columns='SKU').columns)
            for division in st.session_state.df['DIVISION NAME'].unique():
                subdf = st.session_state.df[st.session_state.df['DIVISION NAME']==division].copy()
                skus_in_division = subdf['SKU'].unique()
                suggested_matrix_of_division = suggested_matrix[suggested_matrix['SKU'].isin(skus_in_division)]
                suggested_matrix_totals = suggested_matrix_of_division.drop(columns='SKU').sum(axis=0).to_frame().T
                suggested_matrix_totals['DIVISION NAME'] = division
                suggested_matrix_totals.set_index('DIVISION NAME',inplace=True)
                totals = pd.concat([totals,suggested_matrix_totals])
                
            subset_for_gradient = budget_div.columns.difference(['Total', 'DIVISION NAME'])
            subset_for_format = budget_div.columns.difference(['DIVISION NAME'])
            gradient_args = {
            'cmap': new_cmap,
            'subset': subset_for_gradient,
            'axis': 1
            }
            styled_totals = (totals.sort_index()
                                 .style
                                 .background_gradient(**gradient_args)
                                 .format(fmt_dict)
            )
            st.dataframe(styled_totals, hide_index=False)


        else:
            subdf = st.session_state.df[st.session_state.df['DIVISION NAME'] == division_seleccionada].copy()
            score_cols = ['GT score','ES score','HN TGU score','HN SPS score','RD score']
            
            suggested_matrix = pd.DataFrame(
            0,
            index=subdf.index,
            columns=budget_div.drop(columns='DIVISION NAME').columns
            )

            for col in score_cols:
                country = col.split('score')[0].strip()
                total_scores = subdf[col].sum()
                if total_scores == 0:
                    continue

                relative_col = f'relative {col}'
                subdf[col] = pd.to_numeric(subdf[col], errors='coerce')
                total_scores = pd.to_numeric(total_scores, errors='coerce')
                subdf.loc[:, relative_col] = (subdf[col] / total_scores).fillna(0)
                budget = budget_div[country]

                suggested_col = greedy_rounder(subdf[relative_col], budget)

                # Poner las sugerencias en la columna correspondiente
                suggested_matrix[country] = suggested_col

            # Agregar columna SKU para luego hacer merge/update
            suggested_matrix['SKU'] = subdf['SKU'].values


            # Indexar por SKU para actualizar
            st.session_state.df.set_index('SKU', inplace=True)
            suggested_matrix.set_index('SKU', inplace=True)

            # Actualizar valores solo de las columnas relevantes (países)
            st.session_state.df.update(suggested_matrix)

            st.session_state.df.reset_index(inplace=True)
            
            suggested_matrix['Total'] = suggested_matrix.sum(axis=1)
            suggested_matrix.reset_index(inplace=True)
            gradient_args = {'cmap': new_cmap, 'subset': subset_for_gradient, 'axis': 0}
            styled_suggested = (suggested_matrix.style.background_gradient(**gradient_args))
            
            
            totals = pd.DataFrame([suggested_matrix.sum()])
            totals['DIVISION NAME'] = division_seleccionada
            totals = totals[['DIVISION NAME','GT','ES','RD','HN TGU','HN SPS','Total']]
            
            subset_for_gradient = budget_div.columns.difference(['Total', 'DIVISION NAME'])
            subset_for_format = budget_div.columns.difference(['DIVISION NAME'])
            gradient_args = {
            'cmap': new_cmap,
            'subset': subset_for_gradient,
            'axis': 1
            }
            styled_totals = (totals.sort_index()
                                 .style
                                 .background_gradient(**gradient_args)
                                 .format(fmt_dict)
            )
            st.dataframe(styled_totals, hide_index=True)
            
            st.dataframe(styled_suggested, hide_index=True)
            
    if display_choice == 'Por orden de catálogo':
        st.markdown("#### Estado de la compra")
        
        score_cols = ['GT score','ES score','HN TGU score','HN SPS score','RD score']
        matrixes = {}
        for division in st.session_state.df['DIVISION NAME'].unique():
            subdf = st.session_state.df[st.session_state.df['DIVISION NAME']==division].copy()
            
            suggested_matrix_of_division = pd.DataFrame(
            0,
            index=subdf.index,
            columns=budget_div.drop(columns='DIVISION NAME').columns
            )
            suggested_matrix_of_division['SKU'] = subdf['SKU']
            budget_specific = budget_div[budget_div['DIVISION NAME']==division]  
            
            for col in score_cols:
                country = col.split('score')[0].strip()
                total_scores = subdf[col].sum()

                relative_col = f'relative {col}'
                subdf[col] = pd.to_numeric(subdf[col], errors='coerce')
                total_scores = pd.to_numeric(total_scores, errors='coerce')
                subdf.loc[:, relative_col] = (subdf[col] / total_scores)
                budget = budget_specific[country]
                
                suggested_col = greedy_rounder(subdf[relative_col], budget)

                suggested_matrix_of_division[country] = suggested_col
                
            matrixes[division] = suggested_matrix_of_division
            # Actualizar valores solo de las columnas relevantes (países)
            st.session_state.df = st.session_state.df.set_index('SKU')
            st.session_state.df.update(suggested_matrix_of_division.set_index('SKU'))
            st.session_state.df = st.session_state.df.reset_index()

                
        dfs = [df for df in matrixes.values() if isinstance(df, pd.DataFrame)]
        
        suggested_matrix = pd.concat(dfs)
        
        suggested_matrix['Total'] = suggested_matrix.drop(columns='SKU').sum(axis=1)
        gradient_args = {'cmap': new_cmap, 'subset': subset_for_gradient, 'axis': 0}   
        
        
        totals = pd.DataFrame(columns=suggested_matrix.drop(columns='SKU').columns)
        for division in st.session_state.df['DIVISION NAME'].unique():
            subdf = st.session_state.df[st.session_state.df['DIVISION NAME']==division].copy()
            skus_in_division = subdf['SKU'].unique()
            suggested_matrix_of_division = suggested_matrix[suggested_matrix['SKU'].isin(skus_in_division)]
            suggested_matrix_totals = suggested_matrix_of_division.drop(columns='SKU').sum(axis=0).to_frame().T
            suggested_matrix_totals['DIVISION NAME'] = division
            suggested_matrix_totals.set_index('DIVISION NAME',inplace=True)
            totals = pd.concat([totals,suggested_matrix_totals])
            
        subset_for_gradient = budget_div.columns.difference(['Total', 'DIVISION NAME'])
        subset_for_format = budget_div.columns.difference(['DIVISION NAME'])
        gradient_args = {
        'cmap': new_cmap,
        'subset': subset_for_gradient,
        'axis': 1
        }
        styled_totals = (totals.sort_index()
                             .style
                             .background_gradient(**gradient_args)
                             .format(fmt_dict)
        )
        st.dataframe(styled_totals, hide_index=False)
        
        
# ─────────────────────────────────────────────────────────────────────────────
# 8. Información general de división seleccionada
# ─────────────────────────────────────────────────────────────────────────────
    if display_choice == 'Por división':
        styles_in_division = sorted(df_div['STYLE'].unique())
        
        st.session_state.df.set_index('SKU',inplace=True)
        st.session_state.df.update(suggested_matrix)
        st.session_state.df.reset_index(inplace=True)

        if division_seleccionada == 'Todas las divisiones':
            df_division = st.session_state.df.copy()
        else:
            df_division = st.session_state.df[st.session_state.df['DIVISION NAME'] == division_seleccionada].copy()

        # Información de seleccionados
        sku_unicos = df_division['SKU'].nunique()
        style_unicos = len(styles_in_division)
        gt_ChosenSKUS = df_division[df_division['GT']>0]['SKU'].nunique()
        gt_ChosenStyles = len(df_division[df_division['GT']>0]['STYLE'].unique())
        es_ChosenSKUS = df_division[df_division['ES']>0]['SKU'].nunique()
        es_ChosenStyles = len(df_division[df_division['ES']>0]['STYLE'].unique())
        rd_ChosenSKUS = df_division[df_division['RD']>0]['SKU'].nunique()
        rd_ChosenStyles = len(df_division[df_division['RD']>0]['STYLE'].unique())
        tgu_ChosenSKUS = df_division[df_division['HN TGU']>0]['SKU'].nunique()
        tgu_ChosenStyles = len(df_division[df_division['HN TGU']>0]['STYLE'].unique())
        sps_ChosenSKUS = df_division[df_division['HN SPS']>0]['SKU'].nunique()
        sps_ChosenStyles = len(df_division[df_division['HN SPS']>0]['STYLE'].unique())

        #Información de estilos/SKUs calificados
        gt_GradedSKUS = len(df_division[~df_division['GT score'].isna()]['SKU'].unique())
        es_GradedSKUS = len(df_division[~df_division['ES score'].isna()]['SKU'].unique())
        rd_GradedSKUS = len(df_division[~df_division['RD score'].isna()]['SKU'].unique())
        tgu_GradedSKUS = len(df_division[~df_division['HN TGU score'].isna()]['SKU'].unique())
        sps_GradedSKUS = len(df_division[~df_division['HN SPS score'].isna()]['SKU'].unique())
        gt_GradedStyles = len(df_division[~df_division['GT score'].isna()]['STYLE'].unique())
        es_GradedStyles = len(df_division[~df_division['ES score'].isna()]['STYLE'].unique())
        rd_GradedStyles = len(df_division[~df_division['RD score'].isna()]['STYLE'].unique())
        tgu_GradedStyles = len(df_division[~df_division['HN TGU score'].isna()]['STYLE'].unique())
        sps_GradedStyles = len(df_division[~df_division['HN SPS score'].isna()]['STYLE'].unique())

        col1, col2 = st.columns([7,7])
        with col1:
            st.markdown("#### Productos seleccionados")
            Chosen = pd.DataFrame(columns=['Estilos','SKUs'],index=['Disponible','GT','ES','RD','HN TGU','HN SPS'])
            Chosen.at['Disponible','Estilos'] = style_unicos
            Chosen.at['GT','Estilos'] = gt_ChosenStyles
            Chosen.at['ES','Estilos'] = es_ChosenStyles
            Chosen.at['RD','Estilos'] = rd_ChosenStyles
            Chosen.at['HN TGU','Estilos'] = tgu_ChosenStyles
            Chosen.at['HN SPS','Estilos'] = sps_ChosenStyles
            Chosen.at['Disponible','SKUs'] = sku_unicos
            Chosen.at['GT','SKUs'] = gt_ChosenSKUS
            Chosen.at['ES','SKUs'] = es_ChosenSKUS
            Chosen.at['RD','SKUs'] = rd_ChosenSKUS
            Chosen.at['HN TGU','SKUs'] = tgu_ChosenSKUS
            Chosen.at['HN SPS','SKUs'] = sps_ChosenSKUS
            st.dataframe(Chosen)

        with col2:
            st.markdown("#### Productos calificados")
            Graded = pd.DataFrame(columns=['Estilos','SKUs'],index=['Disponible','GT','ES','RD','HN TGU','HN SPS'])
            Graded.at['Disponible','Estilos'] = style_unicos
            Graded.at['GT','Estilos'] = f"{gt_GradedStyles} calificados, faltan {style_unicos-gt_GradedStyles}"
            Graded.at['ES','Estilos'] = f"{es_GradedStyles} calificados, faltan {style_unicos-es_GradedStyles}"
            Graded.at['RD','Estilos'] = f"{rd_GradedStyles} calificados, faltan {style_unicos-rd_GradedStyles}"
            Graded.at['HN TGU','Estilos'] = f"{tgu_GradedStyles} calificados, faltan {style_unicos-tgu_GradedStyles}"
            Graded.at['HN SPS','Estilos'] = f"{sps_GradedStyles} calificados, faltan {style_unicos-sps_GradedStyles}"
            Graded.at['Disponible','SKUs'] = sku_unicos
            Graded.at['GT','SKUs'] = f"{gt_GradedSKUS} calificados, faltan {sku_unicos-gt_GradedSKUS}"
            Graded.at['ES','SKUs'] = f"{es_GradedSKUS} calificados, faltan {sku_unicos-es_GradedSKUS}"
            Graded.at['RD','SKUs'] = f"{rd_GradedSKUS} calificados, faltan {sku_unicos-rd_GradedSKUS}"
            Graded.at['HN TGU','SKUs'] = f"{tgu_GradedSKUS} calificados, faltan {sku_unicos-tgu_GradedSKUS}"
            Graded.at['HN SPS','SKUs'] = f"{sps_GradedSKUS} calificados, faltan {sku_unicos-sps_GradedSKUS}"
            st.dataframe(Graded)
            

# ─────────────────────────────────────────────────────────────────────────────
# 9. Ventas históricas de la división seleccionada
# ─────────────────────────────────────────────────────────────────────────────
    if display_choice == "Por división":
        if division_seleccionada != "Todas las divisiones":
            col1, col2 = st.columns([1,1])
            with col1:
                ventas_div_hist = ventas_historicas[
                    ventas_historicas['DivisionGenero'].str.startswith(division_seleccionada)
                ]

                if not ventas_div_hist.empty:
                    st.markdown("#### Ventas mensuales por país")

                    pivot_ventas = ventas_div_hist.pivot_table(
                        index='YearMonth',
                        columns='Pais',
                        values='Ventas',
                        aggfunc='sum'
                    ).fillna(0)

                    fig, ax = plt.subplots(figsize=(10, 5))
                    pivot_ventas.plot(ax=ax, marker='o')
                    ax.set_title(f"Ventas mensuales - {division_seleccionada}")
                    ax.set_xlabel("YearMonth")
                    ax.set_ylabel("Ventas")
                    ax.grid(True)
                    ax.legend(title='País', bbox_to_anchor=(1.05, 1), loc='upper left')
                    st.pyplot(fig)
                
                else:
                    st.info(f"No hay datos históricos para la división {division_seleccionada}.")
                

# ─────────────────────────────────────────────────────────────────────────────
# 10. Formularios por estilo
# ─────────────────────────────────────────────────────────────────────────────
    if display_choice == 'Por división':
        if division_seleccionada != 'Todas las divisiones':
            for style, group in grouped:
                st.markdown("---")
                st.markdown(f"### Style: `{style}`")

                row0 = group.iloc[0]
                col_meta1, col_meta2, col_meta3 = st.columns([1, 3, 3])
                with col_meta1:
                    st.markdown(f"**LTA:** ${row0['LTA']}")
                    st.markdown(f"**OUTSOLE:** {row0['OUTSOLE']}")
                    st.markdown(f"**keyItem:** {row0['keyItem']}")
                with col_meta2:
                    st.markdown(f"**FEATURES:** {row0['FEATURES']}")
                    st.markdown(f"**DESCRIPTION:** {row0['DESCRIPTION']}")

                with st.form(key=f"form_{style}"):
                    Ncols = 6
                    cols = st.columns(Ncols)

                    for idx_row, (idx, row) in enumerate(group.iterrows()):
                        col = cols[idx_row % Ncols]
                        sku = row['SKU']
                        product_key = f"{row['DIVISION NAME']}_{sku}_{row['GrupoCatalogo']}"
                        col.write(f"<p style='text-align: center;'>{row['SKU']}</p>", unsafe_allow_html=True)


                        # Inicializar valores desde session_state
                        initial_gt_score = row['GT score']
                        initial_es_score = row['ES score']
                        initial_rd_score = row['RD score']
                        initial_tgu_score = row['HN TGU score']
                        initial_sps_score = row['HN SPS score']

                        # Inputs
                        gt_score = col.text_input("GT puntuación (0-10)", value="" if pd.isna(initial_gt_score) else initial_gt_score, key=f"gt_score_{product_key}")
                        es_score = col.text_input("ES puntuación (0-10)", value="" if pd.isna(initial_es_score) else initial_es_score, key=f"es_score_{product_key}")
                        rd_score = col.text_input("RD puntuación (0-10)", value="" if pd.isna(initial_rd_score) else initial_rd_score, key=f"rd_score_{product_key}")
                        tgu_score = col.text_input("HN TGU puntuación (0-10)", value="" if pd.isna(initial_tgu_score) else initial_tgu_score, key=f"tgu_score_{product_key}")
                        sps_score = col.text_input("HN SPS puntuación (0-10)", value="" if pd.isna(initial_sps_score) else initial_sps_score, key=f"sps_score_{product_key}")

                        # Validaciones (opcional mostrar mensajes)
                        valid_gt_score, gt_score_val = validar_entrada(gt_score)
                        valid_es_score, es_score_val = validar_entrada(es_score)
                        valid_rd_score, rd_score_val = validar_entrada(rd_score)
                        valid_tgu_score, tgu_score_val = validar_entrada(tgu_score)
                        valid_sps_score, sps_score_val = validar_entrada(sps_score)

                        if not valid_gt_score:
                            col.warning("GT debe ser un número entre 0 y 10")
                        if not valid_es_score:
                            col.warning("ES debe ser un número entre 0 y 10")
                        if not valid_rd_score:
                            col.warning("RD debe ser un número entre 0 y 10")
                        if not valid_tgu_score:
                            col.warning("HN TGU debe ser un número entre 0 y 10")
                        if not valid_sps_score:
                            col.warning("HN SPS debe ser un número entre 0 y 10")

                    # Botón para guardar el estilo actual
                    col_guardar = st.columns([1])[0]
                    submitted = col_guardar.form_submit_button("Actualizar")

                    if submitted:
                        st.session_state[f'submitted_success_{style}'] = True

                        for idx_row, (idx, row) in enumerate(group.iterrows()):
                            sku = row['SKU']
                            product_key = f"{row['DIVISION NAME']}_{sku}_{row['GrupoCatalogo']}"

                            gt_score = st.session_state.get(f"gt_score_{product_key}", "")
                            es_score = st.session_state.get(f"es_score_{product_key}", "")
                            rd_score = st.session_state.get(f"rd_score_{product_key}", "")
                            tgu_score = st.session_state.get(f"tgu_score_{product_key}", "")
                            sps_score = st.session_state.get(f"sps_score_{product_key}", "")

                            valid_gt_score, gt_score_val = validar_entrada(gt_score)
                            valid_es_score, es_score_val = validar_entrada(es_score)
                            valid_rd_score, rd_score_val = validar_entrada(rd_score)
                            valid_tgu_score, tgu_score_val = validar_entrada(tgu_score)
                            valid_sps_score, sps_score_val = validar_entrada(sps_score)

                            st.session_state.df.loc[idx, 'GT score'] = gt_score_val
                            st.session_state.df.loc[idx, 'ES score'] = es_score_val
                            st.session_state.df.loc[idx, 'RD score'] = rd_score_val
                            st.session_state.df.loc[idx, 'HN TGU score'] = tgu_score_val
                            st.session_state.df.loc[idx, 'HN SPS score'] = sps_score_val

                        st.rerun()

                # Aplicar cambios si se guardó ese estilo
                if st.session_state.get(f'submitted_success_{style}', False):
                    st.session_state[f'submitted_success_{style}'] = False       
                    st.success("Calificaciones actualizadas correctamente.")
                    
            excel_buffer = io.BytesIO()
            st.session_state.df.to_excel(excel_buffer, index=False, engine='openpyxl')
            excel_buffer.seek(0)

            st.sidebar.download_button(
                label="⬇️ Descargar sugerencias",
                data=excel_buffer,
                file_name="sugerencias.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
                    
        if division_seleccionada == 'Todas las divisiones':
            excel_buffer = io.BytesIO()
            st.session_state.df.to_excel(excel_buffer, index=False, engine='openpyxl')
            excel_buffer.seek(0)

            st.sidebar.download_button(
                label="⬇️ Descargar sugerencias",
                data=excel_buffer,
                file_name="sugerencias.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )

    if display_choice == 'Por orden de catálogo':
        if grupo_seleccionado != 'Todos los grupos':
            for style, group in grouped:
                st.markdown("---")
                st.markdown(f"### Style: `{style}`")

                row0 = group.iloc[0]
                col_meta1, col_meta2, col_meta3 = st.columns([1, 3, 3])
                with col_meta1:
                    st.markdown(f"**LTA:** ${row0['LTA']}")
                    st.markdown(f"**OUTSOLE:** {row0['OUTSOLE']}")
                    st.markdown(f"**keyItem:** {row0['keyItem']}")
                with col_meta2:
                    st.markdown(f"**FEATURES:** {row0['FEATURES']}")
                    st.markdown(f"**DESCRIPTION:** {row0['DESCRIPTION']}")

                with st.form(key=f"form_{style}"):
                    Ncols = 6
                    cols = st.columns(Ncols)

                    for idx_row, (idx, row) in enumerate(group.iterrows()):
                        col = cols[idx_row % Ncols]
                        sku = row['SKU']
                        col.write(row['SKU'])
                        product_key = f"{row['DIVISION NAME']}_{sku}_{row['GrupoCatalogo']}"

                        # Inicializar valores desde session_state
                        initial_gt_score = row['GT score']
                        initial_es_score = row['ES score']
                        initial_rd_score = row['RD score']
                        initial_tgu_score = row['HN TGU score']
                        initial_sps_score = row['HN SPS score']

                        # Inputs
                        gt_score = col.text_input("GT puntuación (0-10)", value="" if pd.isna(initial_gt_score) else initial_gt_score, key=f"gt_score_{product_key}")
                        es_score = col.text_input("ES puntuación (0-10)", value="" if pd.isna(initial_es_score) else initial_es_score, key=f"es_score_{product_key}")
                        rd_score = col.text_input("RD puntuación (0-10)", value="" if pd.isna(initial_rd_score) else initial_rd_score, key=f"rd_score_{product_key}")
                        tgu_score = col.text_input("HN TGU puntuación (0-10)", value="" if pd.isna(initial_tgu_score) else initial_tgu_score, key=f"tgu_score_{product_key}")
                        sps_score = col.text_input("HN SPS puntuación (0-10)", value="" if pd.isna(initial_sps_score) else initial_sps_score, key=f"sps_score_{product_key}")

                        # Validaciones (opcional mostrar mensajes)
                        valid_gt_score, gt_score_val = validar_entrada(gt_score)
                        valid_es_score, es_score_val = validar_entrada(es_score)
                        valid_rd_score, rd_score_val = validar_entrada(rd_score)
                        valid_tgu_score, tgu_score_val = validar_entrada(tgu_score)
                        valid_sps_score, sps_score_val = validar_entrada(sps_score)

                        if not valid_gt_score:
                            col.warning("GT debe ser un número entre 0 y 10")
                        if not valid_es_score:
                            col.warning("ES debe ser un número entre 0 y 10")
                        if not valid_rd_score:
                            col.warning("RD debe ser un número entre 0 y 10")
                        if not valid_tgu_score:
                            col.warning("HN TGU debe ser un número entre 0 y 10")
                        if not valid_sps_score:
                            col.warning("HN SPS debe ser un número entre 0 y 10")

                    # Botón para guardar el estilo actual
                    col_guardar = st.columns([1])[0]
                    submitted = col_guardar.form_submit_button("Actualizar")

                    if submitted:
                        st.session_state[f'submitted_success_{style}'] = True

                        for idx_row, (idx, row) in enumerate(group.iterrows()):
                            sku = row['SKU']
                            product_key = f"{row['DIVISION NAME']}_{sku}_{row['GrupoCatalogo']}"

                            gt_score = st.session_state.get(f"gt_score_{product_key}", "")
                            es_score = st.session_state.get(f"es_score_{product_key}", "")
                            rd_score = st.session_state.get(f"rd_score_{product_key}", "")
                            tgu_score = st.session_state.get(f"tgu_score_{product_key}", "")
                            sps_score = st.session_state.get(f"sps_score_{product_key}", "")

                            valid_gt_score, gt_score_val = validar_entrada(gt_score)
                            valid_es_score, es_score_val = validar_entrada(es_score)
                            valid_rd_score, rd_score_val = validar_entrada(rd_score)
                            valid_tgu_score, tgu_score_val = validar_entrada(tgu_score)
                            valid_sps_score, sps_score_val = validar_entrada(sps_score)

                            st.session_state.df.loc[idx, 'GT score'] = gt_score_val
                            st.session_state.df.loc[idx, 'ES score'] = es_score_val
                            st.session_state.df.loc[idx, 'RD score'] = rd_score_val
                            st.session_state.df.loc[idx, 'HN TGU score'] = tgu_score_val
                            st.session_state.df.loc[idx, 'HN SPS score'] = sps_score_val

                        st.rerun()

                # Aplicar cambios si se guardó ese estilo
                if st.session_state.get(f'submitted_success_{style}', False):
                    st.session_state[f'submitted_success_{style}'] = False
                    st.success("Calificaciones actualizadas correctamente.")
                    
        excel_buffer = io.BytesIO()
        st.session_state.df.to_excel(excel_buffer, index=False, engine='openpyxl')
        excel_buffer.seek(0)

        st.sidebar.download_button(
            label="⬇️ Descargar sugerencias",
            data=excel_buffer,
            file_name="sugerencias.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )


