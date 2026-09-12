
import { useState } from "react";
import {
  MapContainer,
  TileLayer,
  Polyline,
  Marker,
  Popup,
  useMap,
} from "react-leaflet";

import L from "leaflet";
import "leaflet/dist/leaflet.css";
import "./App.css";

// Fix Leaflet marker icons
delete L.Icon.Default.prototype._getIconUrl;

L.Icon.Default.mergeOptions({
  iconRetinaUrl:
    "https://unpkg.com/leaflet@1.9.4/dist/images/marker-icon-2x.png",

  iconUrl:
    "https://unpkg.com/leaflet@1.9.4/dist/images/marker-icon.png",

  shadowUrl:
    "https://unpkg.com/leaflet@1.9.4/dist/images/marker-shadow.png",
});


// ------------------------------------------------------------
// Default values
// Only initial values. User can change them.
// ------------------------------------------------------------

const initialSource = {
  lat: 26.1445,
  lon: 91.7362,
};

const initialDestination = {
  lat: 26.3509,
  lon: 92.6836,
};


// ------------------------------------------------------------
// Fit map to route
// ------------------------------------------------------------

function FitRoute({ route }) {
  const map = useMap();

  if (
    route &&
    route.geometry &&
    route.geometry.coordinates &&
    route.geometry.coordinates.length > 0
  ) {
    const positions = route.geometry.coordinates.map(
      ([lon, lat]) => [lat, lon]
    );

    const bounds = L.latLngBounds(positions);

    map.fitBounds(bounds, {
      padding: [30, 30],
    });
  }

  return null;
}


// ------------------------------------------------------------
// Main App
// ------------------------------------------------------------

function App() {
  const [source, setSource] = useState(initialSource);

  const [destination, setDestination] =
    useState(initialDestination);

  const [route, setRoute] = useState(null);

  const [summary, setSummary] = useState(null);

  const [checkpoints, setCheckpoints] = useState([]);

  const [loading, setLoading] = useState(false);

  const [error, setError] = useState("");


  // ----------------------------------------------------------
  // Find route
  // ----------------------------------------------------------

  const findRoute = async () => {
    setLoading(true);
    setError("");

    try {
      const response = await fetch(
        "http://127.0.0.1:8000/route",
        {
          method: "POST",

          headers: {
            "Content-Type": "application/json",
          },

          body: JSON.stringify({
            source: {
              lat: Number(source.lat),
              lon: Number(source.lon),
            },

            destination: {
              lat: Number(destination.lat),
              lon: Number(destination.lon),
            },
          }),
        }
      );

      if (!response.ok) {
        const errorData = await response.json();

        throw new Error(
          errorData.detail || "Route request failed"
        );
      }

      const data = await response.json();

      setRoute(data.route);

      setSummary(data.summary);

      setCheckpoints(data.checkpoints || []);
    } catch (err) {
      console.error(err);

      setError(
        err.message ||
          "Unable to calculate route."
      );
    } finally {
      setLoading(false);
    }
  };


  return (
    <div className="app">

      {/* HEADER */}
      <header className="header">
        <h1>NER Smart Route Intelligence</h1>

        <p>
          AI-powered risk-aware logistics routing
        </p>
      </header>


      {/* INPUT SECTION */}
      <section className="controls">

        {/* SOURCE */}
        <div className="location-box">

          <h3>Source</h3>

          <div className="inputs">

            <input
              type="number"
              step="any"
              placeholder="Latitude"
              value={source.lat}
              onChange={(e) =>
                setSource({
                  ...source,
                  lat: Number(e.target.value),
                })
              }
            />

            <input
              type="number"
              step="any"
              placeholder="Longitude"
              value={source.lon}
              onChange={(e) =>
                setSource({
                  ...source,
                  lon: Number(e.target.value),
                })
              }
            />

          </div>

        </div>


        {/* DESTINATION */}
        <div className="location-box">

          <h3>Destination</h3>

          <div className="inputs">

            <input
              type="number"
              step="any"
              placeholder="Latitude"
              value={destination.lat}
              onChange={(e) =>
                setDestination({
                  ...destination,
                  lat: Number(e.target.value),
                })
              }
            />

            <input
              type="number"
              step="any"
              placeholder="Longitude"
              value={destination.lon}
              onChange={(e) =>
                setDestination({
                  ...destination,
                  lon: Number(e.target.value),
                })
              }
            />

          </div>

        </div>


        {/* BUTTON */}
        <button
          onClick={findRoute}
          disabled={loading}
        >
          {loading
            ? "Calculating..."
            : "Find Safest Route"}
        </button>

      </section>


      {/* ERROR */}
      {error && (
        <div className="error">
          {error}
        </div>
      )}


      {/* SUMMARY */}
      {summary && (
        <section className="summary">

          <div className="card">
            <span>Distance</span>

            <strong>
              {summary.distance_km} km
            </strong>
          </div>

          <div className="card">
            <span>Dynamic Cost</span>

            <strong>
              {summary.dynamic_cost}
            </strong>
          </div>

          <div className="card">
            <span>Average Hazard</span>

            <strong>
              {(
                summary.average_hazard_probability *
                100
              ).toFixed(1)}
              %
            </strong>
          </div>

          <div className="card">
            <span>Landslide</span>

            <strong>
              {(
                summary.average_landslide_probability *
                100
              ).toFixed(1)}
              %
            </strong>
          </div>

          <div className="card">
            <span>Flood</span>

            <strong>
              {(
                summary.average_flood_probability *
                100
              ).toFixed(1)}
              %
            </strong>
          </div>

          <div className="card">
            <span>Very High Risk</span>

            <strong>
              {summary.very_high_risk_segments}
            </strong>
          </div>

        </section>
      )}


      {/* MAP */}
      <section className="map-wrapper">

        <MapContainer
          center={[26.25, 92.15]}
          zoom={9}
          className="map"
        >

          {/* OSM */}
          <TileLayer
            attribution='&copy; OpenStreetMap contributors'
            url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
          />


          {/* SOURCE */}
          <Marker
            position={[
              source.lat,
              source.lon
            ]}
          >
            <Popup>
              <strong>Source</strong>
              <br />

              Lat: {source.lat}
              <br />

              Lon: {source.lon}
            </Popup>
          </Marker>


          {/* DESTINATION */}
          <Marker
            position={[
              destination.lat,
              destination.lon
            ]}
          >
            <Popup>
              <strong>Destination</strong>
              <br />

              Lat: {destination.lat}
              <br />

              Lon: {destination.lon}
            </Popup>
          </Marker>


          {/* ROUTE */}
          {route &&
            route.geometry &&
            route.geometry.coordinates && (

              <Polyline
                positions={
                  route.geometry.coordinates.map(
                    ([lon, lat]) => [lat, lon]
                  )
                }

                pathOptions={{
                  color: "#2563eb",
                  weight: 6,
                }}
              />

            )}


          {/* CHECKPOINTS */}
          {checkpoints.map((cp) => (

            <Marker
              key={cp.checkpoint_id}
              position={[
                cp.latitude,
                cp.longitude
              ]}
            >

              <Popup>

                <strong>
                  {cp.checkpoint_id}
                </strong>

                <br />

                Distance:
                {" "}
                {cp.distance_from_start_km}
                {" km"}

                <br />

                Hazard:
                {" "}
                {(cp.p_hazard * 100).toFixed(1)}
                %

                <br />

                Landslide:
                {" "}
                {(cp.p_landslide * 100).toFixed(1)}
                %

                <br />

                Flood:
                {" "}
                {(cp.p_flood * 100).toFixed(1)}
                %

                <br />

                Level:
                {" "}
                {cp.hazard_level}

              </Popup>

            </Marker>

          ))}


          {/* FIT ROUTE */}
          {route && (
            <FitRoute route={route} />
          )}

        </MapContainer>

      </section>

    </div>
  );
}

export default App;

