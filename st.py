# import streamlit as st
# import pandas as pd
# import io

# pd.set_option('display.max_columns', None)

# # Streamlit App
# st.title("Tapsi Vendor Orders Processor 🚀")

# st.markdown("""
# Upload multiple CSVs for different cities, select city for each file,  
# enter the Week info ONCE, and download both detailed and summary outputs!
# """)

# uploaded_files = st.file_uploader("Upload CSV files", accept_multiple_files=True, type=['csv'])

# city_options = ['Tehran', 'Mashhad', 'Shiraz']
# city_selection = []

# if uploaded_files:
#     st.subheader("Select City for Each File:")

#     for i, uploaded_file in enumerate(uploaded_files):
#         city = st.selectbox(
#             f"Select city for {uploaded_file.name}",
#             options=city_options,
#             key=f'city_{i}'
#         )
#         city_selection.append(city)

#     st.subheader("Week Information:")
#     week_nu = st.text_input("Enter Week Number (e.g., 04/05)")
#     week_desc = st.text_input("Enter Week Description (e.g., 1-8 Ordibehesht)")

# if st.button("Process Files"):
#     all_dfs = []

#     for i, uploaded_file in enumerate(uploaded_files):
#         if uploaded_file is not None:
#             df = pd.read_csv(uploaded_file)

#             # Basic Cleaning
#             df['City'] = city_selection[i]
#             df['week nu'] = week_nu
#             df['Week'] = week_desc
#             df['vendor website'] = 'https://tapsi.food/vendor/' + df['vendor_code'].astype(str)

#             df.rename(columns={
#                 'vendor_name': 'Vendor title',
#                 'net_order': 'Net orders',
#                 'in_organic_order': 'not organic orders',
#                 'canceled_orders': 'canceled orders',
#                 'business_line': 'Type'
#             }, inplace=True)

#             df = df[['Vendor title', 'week nu', 'Week', 'Net orders', 'not organic orders',
#                      'canceled orders', 'Type', 'vendor website', 'vendor_code', 'City']]

#             all_dfs.append(df)

#     # Merge all
#     final_df = pd.concat(all_dfs).reset_index(drop=True)

#     ###### EXTENDED PROCESSING ######

#     # Cleaning numbers
#     final_df['not organic orders'] = final_df['not organic orders'].replace(',', '', regex=True).astype(int)
#     final_df['Net orders'] = final_df['Net orders'].replace(',', '', regex=True).astype(int)

#     # Create calculated columns
#     final_df['daily_orders'] = round(final_df['Net orders'] / 7, 1)
#     final_df['daily_paid_orders'] = round(final_df['not organic orders'] / 7, 1)
#     final_df['organic share'] = round((final_df['Net orders'] - final_df['not organic orders']) / final_df['Net orders'], 1)



#     ###### DOWNLOAD BUTTONS ######

#     st.success("✅ Processing completed!")

#     st.subheader("Detailed Merged Vendors File")
#     st.dataframe(final_df[['Vendor title', 'week nu', 'Week', 'Net orders', 'not organic orders',
#                      'canceled orders', 'Type', 'vendor website', 'vendor_code', 'City']])

#     csv_buffer_1 = io.StringIO()
#     final_df[['Vendor title', 'week nu', 'Week', 'Net orders', 'not organic orders',
#                      'canceled orders', 'Type', 'vendor website', 'vendor_code', 'City']].to_csv(csv_buffer_1, index=False)
#     st.download_button(
#         label="Download Detailed Vendors CSV",
#         data=csv_buffer_1.getvalue(),
#         file_name="vendors_detailed.csv",
#         mime="text/csv"
#     )

import streamlit as st
import pandas as pd
import numpy as np
import io

pd.set_option('display.max_columns', None)

st.title("Tapsi Vendor Orders Processor 🚀")

st.markdown("""
1. Upload multiple Vendor CSVs (choose city and week once)  
2. Then upload 3 master files  
3. Download 2 final outputs 🎯
""")

# Step 1: Upload Vendor CSVs
if 'vendors_df' not in st.session_state:
    uploaded_files = st.file_uploader("Upload Vendor CSV files", accept_multiple_files=True, type=['csv'])
    city_options = ['Tehran', 'Mashhad', 'Shiraz']

    if uploaded_files:
        st.subheader("City Selection for Each File:")
        city_selection = []

        for i, uploaded_file in enumerate(uploaded_files):
            city = st.selectbox(
                f"Select city for {uploaded_file.name}",
                options=city_options,
                key=f'city_{i}'
            )
            city_selection.append(city)

        st.subheader("Week Information:")
        week_nu = st.text_input("Enter Week Number (e.g., 04/05)")
        week_desc = st.text_input("Enter Week Description (e.g., 1-8 Ordibehesht)")

        if st.button("Process Vendor Files"):
            all_dfs = []
            for i, uploaded_file in enumerate(uploaded_files):
                df = pd.read_csv(uploaded_file)

                df['City'] = city_selection[i]
                df['week nu'] = week_nu
                df['Week'] = week_desc
                df['vendor website'] = 'https://tapsi.food/vendor/' + df['vendor_code'].astype(str)

                df.rename(columns={
                    'vendor_name': 'Vendor title',
                    'net_order': 'Net orders',
                    'in_organic_order': 'not organic orders',
                    'canceled_orders': 'canceled orders',
                    'business_line': 'Type'
                }, inplace=True)

                df = df[['Vendor title', 'week nu', 'Week', 'Net orders', 'not organic orders',
                         'canceled orders', 'Type', 'vendor website', 'vendor_code', 'City']]
                all_dfs.append(df)

            # Save merged vendors_df into session
            st.session_state['vendors_df'] = pd.concat(all_dfs).reset_index(drop=True)

            st.success("✅ Vendors processed and stored successfully!")

            # NOW download the full merged vendors
            csv_buffer = io.StringIO()
            vendors_df = st.session_state['vendors_df']  # <- Correct line
            vendors_df.to_csv(csv_buffer, index=False)
            st.download_button(
                label="Download Vendor Total CSV",
                data=csv_buffer.getvalue(),
                file_name="vendor_total.csv",
                mime="text/csv"
            )
            # st.rerun()  # Refresh to move to next step

# Step 2: Upload Master Files
else:
    st.success("✅ Vendors already processed!")

    st.subheader("Upload Master Files:")

    df_uploaded = st.file_uploader("Upload 'Vendors per city-Weekly - Total.csv'", type='csv')
    kpi_uploaded = st.file_uploader("Upload 'Vendors KPI.xlsx'", type='xlsx')
    tf_uploaded = st.file_uploader("Upload 'Competitor List-March 2024 - List.csv'", type='csv')

    if df_uploaded and kpi_uploaded and tf_uploaded:
        vendors_df = st.session_state['vendors_df']

        # Load additional data
        df = pd.read_csv(df_uploaded)
        kpi = pd.read_excel(kpi_uploaded, sheet_name='Vendors KPI', skiprows=3, usecols='B:Z')
        tf = pd.read_csv(tf_uploaded, skiprows=1)

        df1 = df.copy()

        # Process tf
        tf = tf.loc[tf['Competitor'] == 'Ofood', ['Vendor Code', 'Updated Wishlist Grade', 'Decile', 'Contract Type', 'Vendor id', 'Current Status']]
        tf['Vendor id'] = tf['Vendor id'].astype(str).str.strip().str.replace(r'\D', '', regex=True)
        tf['Vendor id'] = pd.to_numeric(tf['Vendor id'], errors='coerce')
        tf = tf.merge(kpi[['Vendor ID', 'Vendor Current Status']], how='left', left_on='Vendor id', right_on='Vendor ID')
        tf.rename(columns={'Current Status': 'TapsiFood Status'}, inplace=True)

        # Clean df
        df = df[df['Net orders'].notnull()]
        df['Net orders'] = df['Net orders'].replace(',', '', regex=True).astype(int)
        df['week nu'] = df['week nu'].replace('/', '.', regex=True).astype(float)
        df['not organic orders'] = df['not organic orders'].replace(',', '', regex=True).astype(int)

        df['organic share'] = round((df['Net orders']-df['not organic orders'])/df['Net orders'], 3)
        df['daily orders'] = np.ceil(df['Net orders']/7)
        df['daily organic orders'] = np.ceil((df['Net orders']-df['not organic orders'])/7)

        def custom_range_daily(order_count):
            if order_count == 0: return "0"
            if order_count <= 6: return f"{int((np.ceil(order_count/2)-1)*2+1)}-{int((np.ceil(order_count/2)-1)*2+2)}"
            if order_count <= 10: return "7-10"
            if order_count <= 20: return f"{int((np.ceil(order_count/5)-1)*5+1)}-{int((np.ceil(order_count/5)-1)*5+5)}"
            if order_count <= 100: return f"{int((np.ceil(order_count/10)-1)*10+1)}-{int((np.ceil(order_count/10)-1)*10+10)}"
            else: return f"{int((np.floor(order_count/50))*50+1)}-{int((np.floor(order_count/50))*50+50)}"

        df = df.sort_values(by=['vendor_code','week nu'])
        df['Food'] = np.where(df['Type'].isin(['Restaurant', 'Ice Cream and Juice Shop', 'Cafe']), 'Yes', 'No')
        df['Rank'] = df.groupby(['City', 'Week', 'Food'])['Net orders'].rank(ascending=False, method='min')
        df['is_first_week_with_sales'] = df.groupby('vendor_code')['week nu'].transform(lambda x: x.gt(0).idxmax()) == df.index
        df['More than 3 orders'] = np.where(df['Net orders'] > 3, 'Yes', 'No')
        df['daily orders'] = df['daily orders'].apply(custom_range_daily)
        df['organic share'] = df['organic share'].fillna(0)

        df = df[['Vendor title', 'week nu', 'Week', 'Type', 'daily orders', 'organic share', 'vendor website', 
                 'vendor_code', 'City', 'More than 3 orders', 'is_first_week_with_sales', 'Food', 'Rank', 'daily organic orders']]

        def binning_function(group):
            unique_values = np.sort(group['organic share'].unique())
            if len(unique_values) > 1:
                bins = pd.qcut(unique_values, q=min(5, len(unique_values)), labels=['Very Low', 'Low', 'Medium', 'High', 'Very High'])
                bin_mapping = dict(zip(unique_values, bins))
                group['organic share'] = group['organic share'].map(bin_mapping)
            else:
                group['organic share'] = 'No Bin'
            return group

        df = df.groupby(['week nu', 'Food', 'City'], group_keys=False).apply(binning_function)

        final = df.merge(tf, how='left', left_on='vendor_code', right_on='Vendor Code')
        final.replace(['-', '_'], np.nan, inplace=True)
        final['Decile'] = final['Decile'].astype(float)
        final.loc[final['Vendor id'].isnull(), 'Vendor Current Status'] = 'Not-have'
        final.loc[final['Vendor Current Status'].isnull(), 'Vendor Current Status'] = 'Without-status'
        final_output = final.drop(columns=['Vendor Code', 'Vendor ID', 'Rank']).sort_values(by=['City','vendor_code','week nu'])

        # Now process vendors_df for summary
        df1['not organic orders'] = df1['not organic orders'].replace(',', '', regex=True).astype(int)
        df1['Net orders'] = df1['Net orders'].replace(',', '', regex=True).astype(int)
        df1['daily_orders'] = round(df1['Net orders']/7,1)
        df1['daily_paid_orders'] = round(df1['not organic orders']/7,1)
        df1['organic share'] = round((df1['Net orders']-df1['not organic orders'])/df1['Net orders'],1)

        mapper = {
            'Bakery': '3-Bakery',
            'Ice Cream and Juice Shop': '1-Food',
            'Cafe': '1-Food',
            'Meat Shop': '2-Non-Food',
            'Fruit Shop': '2-Non-Food',
            'Restaurant': '1-Food',
            'Pastry': '2-Non-Food'
        }
        df1['food_type'] = df1['Type'].map(mapper)

        df1['City'] = df1['City'].replace({'Tehran': '1-Tehran', 'Mashhad': '2-Mashhad', 'Shiraz': '3-Shiraz'})

        first_appearance = df1.groupby('vendor_code')['week nu'].min().reset_index()
        first_appearance.columns = ['vendor_code', 'First_Week']
        df1 = df1.merge(first_appearance, on='vendor_code')
        df1['Acquisition_Status'] = df1.apply(lambda row: 1 if row['week nu'] == row['First_Week'] else 0, axis=1)

        ord = df1.groupby(['week nu','City','food_type']).agg(
            daily_orders=('daily_orders','sum'),
            daily_paid_orders=('daily_paid_orders','sum'),
            vendors=('Vendor title','count'),
            new_vendors=('Acquisition_Status','sum')
        ).reset_index()

        ord['organic_share'] = round(1-(ord['daily_paid_orders']/ord['daily_orders']),2)
        ord['daily_organic_orders'] = ord['daily_orders']-ord['daily_paid_orders']

        output_summary = ord[['food_type','City','week nu','daily_orders','daily_paid_orders','daily_organic_orders','vendors','new_vendors','organic_share']]\
                         .sort_values(by=['food_type','City','week nu'])

        st.success("✅ Final Outputs Ready!")

        # Download buttons
        tf_buffer = io.BytesIO()
        with pd.ExcelWriter(tf_buffer, engine='xlsxwriter') as writer:
            final_output.to_excel(writer, index=False)
        st.download_button(
            label="Download TF Vendors Excel",
            data=tf_buffer.getvalue(),
            file_name="TF_Vendors.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

        csv_buffer = io.StringIO()
        output_summary.to_csv(csv_buffer, index=False)
        st.download_button(
            label="Download Output Summary CSV",
            data=csv_buffer.getvalue(),
            file_name="output_summary.csv",
            mime="text/csv"
        )
