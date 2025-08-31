from flask import Flask, request, jsonify, render_template, redirect, url_for, flash
from flask_sqlalchemy import SQLAlchemy
from flask_socketio import SocketIO
from flask_cors import CORS
from flask_login import LoginManager, UserMixin, login_user, logout_user, login_required, current_user
from flask_migrate import Migrate
from datetime import datetime, timedelta
import json
import os
from dotenv import load_dotenv
import logging
import requests
import schedule
import eventlet
import tensorflow as tf
from twilio.rest import Client
from twilio.base.exceptions import TwilioRestException
from werkzeug.security import generate_password_hash, check_password_hash

# Load environment variables
load_dotenv()

app = Flask(__name__)
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'coastal-alert-secret-key-2025')
app.config['SQLALCHEMY_DATABASE_URI'] = os.environ.get('DATABASE_URL', 'sqlite:///coastal_alerts.db')
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['TOMORROW_API_KEY'] = os.environ.get('TOMORROW_API_KEY', 'your_api_key_here')
app.config['TWILIO_SID'] = os.environ.get('TWILIO_SID', 'your_twilio_sid')
app.config['TWILIO_TOKEN'] = os.environ.get('TWILIO_TOKEN', 'your_twilio_token')
app.config['TWILIO_PHONE'] = os.environ.get('TWILIO_PHONE', '+1234567890')

# Initialize extensions
db = SQLAlchemy(app)
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='eventlet')
CORS(app)
login_manager = LoginManager(app)
login_manager.login_view = 'login'
migrate = Migrate(app, db)

# Setup logging
logging.basicConfig(level=logging.INFO)
app.logger.setLevel(logging.INFO)

# Initialize Twilio client
try:
    twilio_client = Client(app.config['TWILIO_SID'], app.config['TWILIO_TOKEN'])
except Exception as e:
    app.logger.warning(f"Twilio not configured: {str(e)}")
    twilio_client = None

# Database Models
class Alert(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    threat_type = db.Column(db.String(64), index=True, nullable=False)
    location = db.Column(db.String(128))
    latitude = db.Column(db.Float, nullable=False)
    longitude = db.Column(db.Float, nullable=False)
    risk_level = db.Column(db.String(16), index=True, nullable=False)
    timestamp = db.Column(db.DateTime, index=True, default=datetime.utcnow)
    details = db.Column(db.Text, nullable=True)
    value = db.Column(db.Float, default=0.0)
    acknowledged = db.Column(db.Boolean, default=False)

    def to_dict(self):
        return {
            'id': self.id,
            'threat_type': self.threat_type,
            'location': self.location,
            'latitude': self.latitude,
            'longitude': self.longitude,
            'risk_level': self.risk_level,
            'timestamp': self.timestamp.isoformat() + 'Z',
            'details': self.details,
            'value': self.value,
            'acknowledged': self.acknowledged
        }

class User(db.Model, UserMixin):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(100), unique=True, nullable=False)
    password_hash = db.Column(db.String(128), nullable=False)
    phone_number = db.Column(db.String(20), unique=True, nullable=False)
    name = db.Column(db.String(100))
    role = db.Column(db.String(50), default='user')

    def set_password(self, password):
        self.password_hash = generate_password_hash(password)

    def check_password(self, password):
        return check_password_hash(self.password_hash, password)

# User loader for Flask-Login
@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

# Deep Learning Model Functions
def load_dl_model():
    try:
        # In production, load your trained model
        # model = tf.keras.models.load_model('model/weather_model.h5')
        # app.logger.info("Loaded pre-trained DL model")
        
        # For demo, create a simple model
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(128, activation='relu', input_shape=(6,)),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(3, activation='softmax')
        ])
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        app.logger.warning("Using demo DL model")
    except Exception as e:
        app.logger.error(f"Model loading error: {str(e)}")
        model = None
    return model

dl_model = load_dl_model()

def get_risk_level_from_dl(data):
    """Use DL model to predict risk level"""
    try:
        if dl_model is None:
            return get_risk_level_from_ai(data)
            
        features = [
            data.get('wind_speed', 0) / 100,
            data.get('sea_level', 0) / 10,
            data.get('precipitation', 0) / 100,
            data.get('temperature', 20) / 50,
            data.get('humidity', 50) / 100,
            data.get('wave_height', 0) / 10
        ]
        prediction = dl_model.predict([features], verbose=0)[0]
        labels = ['Watch', 'Warning', 'Red Alert']
        return labels[prediction.argmax()]
    except Exception as e:
        app.logger.error(f"DL prediction error: {str(e)}")
        return get_risk_level_from_ai(data)

def get_risk_level_from_ai(data):
    """Fallback AI-based risk assessment"""
    threat_type = data.get('threat_type', '').lower()
    value = data.get('value', 0)
    wave_height = data.get('wave_height', 0)
    
    if 'tsunami' in threat_type or wave_height > 5:
        return 'Red Alert' if wave_height > 7 else 'Warning' if wave_height > 3 else 'Watch'
    elif 'storm' in threat_type or 'hurricane' in threat_type or 'cyclone' in threat_type:
        return 'Red Alert' if value > 7 else 'Warning' if value > 4 else 'Watch'
    elif 'pollution' in threat_type:
        return 'Red Alert' if value > 150 else 'Warning' if value > 100 else 'Watch'
    elif 'erosion' in threat_type or 'flood' in threat_type:
        return 'Red Alert' if value > 50 else 'Warning' if value > 25 else 'Watch'
    else:
        return 'Red Alert' if value > 80 else 'Warning' if value > 50 else 'Watch'

# Weather Data Functions
def fetch_weather_data():
    """Fetch weather data from Tomorrow.io API"""
    if not app.config['TOMORROW_API_KEY'] or app.config['TOMORROW_API_KEY'] == 'your_api_key_here':
        app.logger.warning("Tomorrow.io API key not configured, skipping weather fetch")
        return
        
    locations = [
        {'name': 'Surat Coast', 'lat': 21.1702, 'lon': 72.8311},
        {'name': 'Kandla Port', 'lat': 22.3072, 'lon': 68.9673},
        {'name': 'Valsad Coast', 'lat': 20.2809, 'lon': 72.9169},
        {'name': 'Mundra Port', 'lat': 22.8395, 'lon': 69.7116},
        {'name': 'Dwarka', 'lat': 22.2394, 'lon': 68.9678}
    ]
    
    url = "https://api.tomorrow.io/v4/weather/realtime"
    params = {
        'apikey': app.config['TOMORROW_API_KEY'],
        'fields': ['windSpeed', 'precipitationIntensity', 'seaLevelPressure', 'temperature', 'humidity'],
        'units': 'metric'
    }
    headers = {'accept': 'application/json'}

    for loc in locations:
        params['location'] = f"{loc['lat']},{loc['lon']}"
        try:
            response = requests.get(url, params=params, headers=headers, timeout=10)
            response.raise_for_status()
            data = response.json()['data']['values']
            
            # Create alert based on weather conditions
            alert_data = {
                'threat_type': 'Weather Monitoring',
                'location': loc['name'],
                'latitude': loc['lat'],
                'longitude': loc['lon'],
                'value': max(data.get('windSpeed', 0), data.get('precipitationIntensity', 0) * 10),
                'wind_speed': data.get('windSpeed', 0),
                'sea_level': data.get('seaLevelPressure', 1013) / 100,
                'precipitation': data.get('precipitationIntensity', 0),
                'temperature': data.get('temperature', 25),
                'humidity': data.get('humidity', 60),
                'wave_height': data.get('windSpeed', 0) / 10  # Estimate wave height from wind speed
            }
            
            # Only create alert if significant weather event
            if alert_data['value'] > 30:  # Threshold for creating alert
                process_weather_alert(alert_data)
                
        except requests.RequestException as e:
            app.logger.error(f"Failed to fetch weather data for {loc['name']}: {str(e)}")

def process_weather_alert(data):
    """Process and store weather alert"""
    risk_level = get_risk_level_from_dl(data)
    
    alert = Alert(
        threat_type=data.get('threat_type', 'Weather Event'),
        location=data.get('location', 'Coastal Area'),
        latitude=data['latitude'],
        longitude=data['longitude'],
        risk_level=risk_level,
        details=json.dumps(data),
        value=data.get('value', 0)
    )
    
    db.session.add(alert)
    db.session.commit()
    
    # Emit to all connected clients
    socketio.emit('new_alert', alert.to_dict())
    
    app.logger.info(f"New weather alert: {alert.threat_type} - {alert.risk_level} at {alert.location}")
    
    # Send SMS for critical alerts
    if risk_level == 'Red Alert' and twilio_client:
        send_sms_notifications(alert)

def send_sms_notifications(alert):
    """Send SMS notifications for critical alerts"""
    if not twilio_client:
        app.logger.warning("Twilio not configured, skipping SMS")
        return
        
    users = User.query.all()
    for user in users:
        try:
            message = twilio_client.messages.create(
                body=f"🚨 Red Alert: {alert.threat_type} at {alert.location}. Immediate action required! Issued: {alert.timestamp.strftime('%H:%M')}",
                from_=app.config['TWILIO_PHONE'],
                to=user.phone_number
            )
            app.logger.info(f"SMS sent to {user.phone_number}: {message.sid}")
        except TwilioRestException as e:
            app.logger.error(f"Failed to send SMS to {user.phone_number}: {str(e)}")

# Routes
@app.route('/')
def dashboard():
    """Main dashboard"""
    return render_template('dashboard.html')

@app.route('/alerts')
def alerts_page():
    """Alerts management page"""
    return render_template('alerts.html')

@app.route('/monitoring')
def monitoring_page():
    """Live monitoring page"""
    return render_template('monitoring.html')

@app.route('/analytics')
def analytics_page():
    """Analytics and reports page"""
    return render_template('analytics.html')

@app.route('/settings')
@login_required
def settings_page():
    """User settings page"""
    return render_template('settings.html')

# API Routes
@app.route('/api/current_user', methods=['GET'])
def current_user_api():
    if current_user.is_authenticated:
        return jsonify({
            'username': current_user.username,
            'name': current_user.name,
            'role': current_user.role
        })
    return jsonify({'username': None})

@app.route('/api/alerts', methods=['GET'])
def get_alerts():
    """Get alerts with optional filtering"""
    threat_filter = request.args.get('type')
    risk_filter = request.args.get('risk')
    limit = int(request.args.get('limit', 50))
    
    query = Alert.query
    if threat_filter:
        query = query.filter(Alert.threat_type.ilike(f'%{threat_filter}%'))
    if risk_filter:
        query = query.filter(Alert.risk_level == risk_filter)
    
    alerts = query.order_by(Alert.timestamp.desc()).limit(limit).all()
    return jsonify([alert.to_dict() for alert in alerts])

@app.route('/api/alerts/<int:alert_id>/acknowledge', methods=['POST'])
@login_required
def acknowledge_alert(alert_id):
    """Acknowledge an alert"""
    alert = Alert.query.get_or_404(alert_id)
    alert.acknowledged = True
    db.session.commit()
    
    socketio.emit('alert_acknowledged', {'alert_id': alert_id})
    app.logger.info(f"Alert {alert_id} acknowledged by {current_user.username}")
    
    return jsonify({'status': 'success', 'message': 'Alert acknowledged'})

@app.route('/api/ingest', methods=['POST'])
def ingest_data():
    """Ingest new alert data"""
    try:
        data = request.get_json()
        if not data or 'latitude' not in data or 'longitude' not in data:
            return jsonify({'error': 'Missing required fields: latitude, longitude'}), 400
        
        risk_level = get_risk_level_from_dl(data)
        
        alert = Alert(
            threat_type=data.get('threat_type', 'Unknown Threat'),
            location=data.get('location', 'Coastal Area'),
            latitude=float(data['latitude']),
            longitude=float(data['longitude']),
            risk_level=risk_level,
            details=json.dumps(data),
            value=float(data.get('value', 0))
        )
        
        db.session.add(alert)
        db.session.commit()
        
        # Emit to all connected clients
        socketio.emit('new_alert', alert.to_dict())
        
        # Send SMS for critical alerts
        if risk_level == 'Red Alert' and twilio_client:
            send_sms_notifications(alert)
        
        app.logger.info(f"New alert created: {alert.threat_type} - {alert.risk_level} at {alert.location}")
        
        return jsonify({
            'status': 'success',
            'alert_id': alert.id,
            'risk_level': risk_level,
            'message': 'Alert processed and broadcasted'
        }), 201
        
    except Exception as e:
        app.logger.error(f"Error processing alert: {str(e)}")
        return jsonify({'error': 'Failed to process alert data'}), 500

@app.route('/api/stats', methods=['GET'])
def get_stats():
    """Get system statistics"""
    try:
        total_alerts = Alert.query.count()
        red_alerts = Alert.query.filter_by(risk_level='Red Alert').count()
        warnings = Alert.query.filter_by(risk_level='Warning').count()
        watches = Alert.query.filter_by(risk_level='Watch').count()
        
        yesterday = datetime.utcnow() - timedelta(days=1)
        recent_alerts = Alert.query.filter(Alert.timestamp >= yesterday).count()
        
        # Get alerts by location
        location_stats = db.session.query(
            Alert.location, 
            db.func.count(Alert.id).label('count')
        ).group_by(Alert.location).all()
        
        return jsonify({
            'total_alerts': total_alerts,
            'red_alerts': red_alerts,
            'warnings': warnings,
            'watches': watches,
            'recent_activity': recent_alerts,
            'location_stats': [{'location': loc, 'count': count} for loc, count in location_stats],
            'active_users': User.query.count()
        })
    except Exception as e:
        app.logger.error(f'Error fetching stats: {str(e)}')
        return jsonify({'error': 'Failed to fetch statistics'}), 500

# Authentication Routes
@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        data = request.form
        username = data.get('username')
        password = data.get('password')
        phone_number = data.get('phone_number')
        name = data.get('name')
        
        # Validation
        if User.query.filter_by(username=username).first():
            flash('Username already exists', 'error')
            return render_template('auth.html', form_type='register')
        
        if User.query.filter_by(phone_number=phone_number).first():
            flash('Phone number already registered', 'error')
            return render_template('auth.html', form_type='register')
        
        # Create new user
        user = User(username=username, phone_number=phone_number, name=name)
        user.set_password(password)
        db.session.add(user)
        db.session.commit()
        
        login_user(user)
        flash('Registration successful!', 'success')
        return redirect(url_for('dashboard'))
    
    return render_template('auth.html', form_type='register')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        data = request.form
        username = data.get('username')
        password = data.get('password')
        
        user = User.query.filter_by(username=username).first()
        
        if user and user.check_password(password):
            login_user(user)
            flash('Login successful!', 'success')
            return redirect(url_for('dashboard'))
        else:
            flash('Invalid credentials', 'error')
    
    return render_template('auth.html', form_type='login')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    flash('Logged out successfully', 'info')
    return redirect(url_for('login'))

# Tomorrow.io Webhook
@app.route('/api/webhook/tomorrow', methods=['POST'])
def tomorrow_webhook():
    """Handle Tomorrow.io webhook data"""
    try:
        data = request.get_json()
        location = data.get('location', {})
        values = data.get('data', {})
        
        alert_data = {
            'threat_type': values.get('eventType', 'Weather Event'),
            'location': location.get('name', 'Coastal Area'),
            'latitude': float(location.get('lat', 21.1702)),
            'longitude': float(location.get('lon', 72.8311)),
            'value': max(
                values.get('windSpeed', 0), 
                values.get('precipitationIntensity', 0) * 10, 
                values.get('waveHeight', 0) * 5
            ),
            'wind_speed': values.get('windSpeed', 0),
            'sea_level': values.get('seaLevelPressure', 0) / 100,
            'precipitation': values.get('precipitationIntensity', 0),
            'temperature': values.get('temperature', 20),
            'humidity': values.get('humidity', 50),
            'wave_height': values.get('waveHeight', 0)
        }
        
        process_weather_alert(alert_data)
        return jsonify({'status': 'success'}), 200
        
    except Exception as e:
        app.logger.error(f"Webhook error: {str(e)}")
        return jsonify({'error': 'Failed to process webhook'}), 500

# WebSocket Event Handlers
@socketio.on('connect')
def handle_connect():
    app.logger.info(f'Client connected: {request.sid}')
    try:
        # Send latest alerts to the connecting client
        alerts = Alert.query.order_by(Alert.timestamp.desc()).limit(50).all()
        socketio.emit('alerts_update', [alert.to_dict() for alert in alerts], room=request.sid)
    except Exception as e:
        app.logger.error(f"Error sending alerts on connect: {str(e)}")

@socketio.on('disconnect')
def handle_disconnect():
    app.logger.info(f'Client disconnected: {request.sid}')

@socketio.on('request_alerts')
def handle_request_alerts():
    """Handle client request for latest alerts"""
    try:
        alerts = Alert.query.order_by(Alert.timestamp.desc()).limit(50).all()
        socketio.emit('alerts_update', [alert.to_dict() for alert in alerts], room=request.sid)
    except Exception as e:
        app.logger.error(f"Error handling request_alerts: {str(e)}")

# Background Scheduler
def run_scheduler():
    """Background task for scheduled operations"""
    schedule.every(5).minutes.do(fetch_weather_data)
    schedule.every(1).hours.do(cleanup_old_alerts)
    
    while True:
        schedule.run_pending()
        eventlet.sleep(60)

def cleanup_old_alerts():
    """Clean up alerts older than 7 days"""
    try:
        cutoff_date = datetime.utcnow() - timedelta(days=7)
        old_alerts = Alert.query.filter(Alert.timestamp < cutoff_date).all()
        
        for alert in old_alerts:
            db.session.delete(alert)
        
        db.session.commit()
        app.logger.info(f"Cleaned up {len(old_alerts)} old alerts")
    except Exception as e:
        app.logger.error(f"Error cleaning up alerts: {str(e)}")

# Demo data creation
def create_demo_data():
    """Create demo alerts and users if database is empty"""
    if Alert.query.count() == 0:
        demo_alerts = [
            Alert(
                threat_type='High Tide Warning',
                location='Mundra Port',
                latitude=22.8395,
                longitude=69.7116,
                risk_level='Warning',
                value=65.0,
                details='{"tide_height": 2.8, "expected_peak": "18:30"}'
            ),
            Alert(
                threat_type='Cyclone Activity',
                location='Bhavnagar',
                latitude=21.7645,
                longitude=72.1519,
                risk_level='Red Alert',
                value=85.0,
                details='{"wind_speed": 120, "pressure_drop": 15}'
            ),
            Alert(
                threat_type='Coastal Erosion',
                location='Veraval',
                latitude=20.9077,
                longitude=70.3665,
                risk_level='Watch',
                value=35.0,
                details='{"erosion_rate": 1.2, "affected_area": 500}'
            ),
            Alert(
                threat_type='Water Quality Alert',
                location='Hazira',
                latitude=21.1116,
                longitude=72.6000,
                risk_level='Warning',
                value=75.0,
                details='{"pollution_level": 125, "source": "industrial"}'
            )
        ]
        
        for alert in demo_alerts:
            db.session.add(alert)
        
        db.session.commit()
        app.logger.info("Demo alerts created")
    
    if User.query.count() == 0:
        demo_users = [
            User(username='admin', phone_number='+919876543210', name='Admin User', role='admin'),
            User(username='operator', phone_number='+919876543211', name='System Operator', role='operator'),
            User(username='viewer', phone_number='+919876543212', name='Alert Viewer', role='user')
        ]
        
        for user in demo_users:
            user.set_password('password123')
            db.session.add(user)
        
        db.session.commit()
        app.logger.info("Demo users created (password: password123)")

# Initialize database
with app.app_context():
    try:
        db.create_all()
        create_demo_data()
        app.logger.info("Database initialized successfully")
    except Exception as e:
        app.logger.error(f"Database initialization error: {str(e)}")

if __name__ == '__main__':
    # Start background scheduler
    eventlet.spawn(run_scheduler)
    
    # Run the application
    socketio.run(
        app, 
        debug=True, 
        host='0.0.0.0', 
        port=5000,
        allow_unsafe_werkzeug=True
    )