import requests
import xml.etree.ElementTree as ET
import pandas as pd

class BMSChillerClient:
    def __init__(self, url, username, password):
        self.url = url
        self.username = username
        self.password = password
        self.headers = {'Content-Type': 'application/xml'}

    def build_query_xml(self, dev_addr, var_codes, start_time, frequency="30m", period="W"):
        """Construye el XML según el formato de tu imagen"""
        # Inicio del paquete
        xml = f'<requests>\n'
        xml += f'    <login userName="{self.username}" password="{self.password}" />\n'
        xml += f'    <request type="getHistoricalValues" frequency="{frequency}" starttime="{start_time}" TimePeriod="{period}" language="EN_en">\n'
        
        # Dispositivo y variables
        xml += f'        <dev devaddr="{dev_addr}">\n'
        for code in var_codes:
            xml += f'            <var code="{code}"/>\n'
        
        # Cierre del paquete
        xml += '        </dev>\n'
        xml += '    </request>\n'
        xml += '</requests>'
        return xml

    def get_historical_data(self, xml_payload):
        """Envía la consulta y procesa la respuesta XML"""
        try:
            response = requests.post(self.url, data=xml_payload, headers=self.headers, timeout=30)
            response.raise_for_status()
            return self._parse_to_dataframe(response.text)
        except Exception as e:
            print(f"Error en la comunicación: {e}")
            return None

    def _parse_to_dataframe(self, xml_string):
        """Convierte la respuesta XML en un DataFrame de Pandas para ML"""
        root = ET.fromstring(xml_string)
        all_records = []

        # Navegar por la estructura: responses -> response -> dev -> var -> val
        for dev in root.findall('.//dev'):
            dev_addr = dev.get('devaddr')
            for var in dev.findall('var'):
                var_code = var.get('code')
                for val in var.findall('val'):
                    all_records.append({
                        'timestamp': val.get('time'),
                        'variable': var_code,
                        'value': val.get('value'),
                        'dev_addr': dev_addr
                    })
        
        df = pd.DataFrame(all_records)
        if not df.empty:
            # Limpieza básica para ML: convertir valor a numérico y timestamp a datetime
            df['value'] = pd.to_numeric(df['value'], errors='coerce')
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            # Pivotar para tener una columna por variable (ideal para predicción)
            df = df.pivot(index='timestamp', columns='variable', values='value')
        
        return df

# --- CONFIGURACIÓN Y EJECUCIÓN ---

# Datos del servidor y credenciales (según imagen 1)
URL_BMS = "http://TU_IP_SERVER/queryXML" 
USER = "user"
PASS = "password"

client = BMSChillerClient(URL_BMS, USER, PASS)

# Definir qué queremos consultar (según imagen 1 y 2)
dispositivo = "3.033"
variables = ["code1", "code2"] # Ej: "Temp_Retorno", "Presion_Aceite"
fecha_inicio = "2019-11-15"

# 1. Generar XML
query = client.build_query_xml(dispositivo, variables, fecha_inicio)

# 2. Consultar y obtener DataFrame
df_historicos = client.get_historical_data(query)

if df_historicos is not None:
    print("📊 Datos recuperados para el modelo ML:")
    print(df_historicos.head())
    
    # 3. Guardar para entrenamiento posterior
    # df_historicos.to_csv("datos_chiller.csv")
